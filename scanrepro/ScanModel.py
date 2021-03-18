import numpy as np
import torch
import torch.nn as nn
from . import debug as db
import torch.nn.functional as F
from collections import OrderedDict
from scanrepro import SimCLRModel
import scancuda
from torchvision import models
import pytorch_lightning as pl

class SCANModelPT(nn.Module):
    def __init__(
        self,
        model: SimCLRModel.SIMCLRModelPT,
        in_size: int = 512,
        n_classes: int = 10,
        **kwargs
    ):
        super().__init__()
        self.model = model.resnet_feat
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_size, n_classes),
            torch.nn.Softmax(dim=1)
        )

    def setSelfLabel(self):
        self.nn = self.nn[:-1]

    def forward_impl(self, x):
        bs = x.shape[0]
        h = self.model(x).view(bs, -1)
        return self.nn(h)

    def forward(self, x):
        return self.forward_impl(x)

    def entropyLoss(self, p):
        p_mean = p.mean(0).clamp(min=1e-8)
        log_p_mean = p_mean.log()
        entropy = p_mean * log_p_mean
        return -entropy.sum()

    def SCANLoss(self, p, p_neighbor, p_idx, p_idx_neighbor, nearest, entropy_lambda):
        entropy = self.entropyLoss(p)
        prob_grid = p @ p_neighbor.transpose(0, 1)
        maskout = torch.zeros([*prob_grid.shape], device=prob_grid.device, dtype=torch.int32)
        #
        # Create a mask with matched pairs.
        scancuda.SCAN_NN_Mask_Fill(p_idx_neighbor, nearest, maskout)
        dotloss = prob_grid
        dotloss = dotloss.log()[maskout.bool()]
        dotloss = -dotloss.mean()
        return dotloss, entropy

    def selfLabelLoss(self, logits_orig, logits_tf):
        p = F.softmax(logits_orig, dim=1)
        p_max, max_idx = p.max(dim=1)
        masktmp = p_max > 0.99
        #
        # Keep only the instances with prob > threshold.
        logits_orig_tmp = logits_orig[masktmp]
        max_idx = max_idx[masktmp].detach()
        #
        # Weights for cross-entropy.
        idx, cnts = max_idx.unique(return_counts=True)
        sm_weights = torch.ones([logits_orig_tmp.shape[-1]], device=logits_orig_tmp.device)
        weights= 1/(cnts / max_idx.size(0))
        sm_weights[idx] = weights
        CE_loss = F.cross_entropy(logits_tf[masktmp], max_idx, weight=sm_weights, reduction='mean')
        return CE_loss
class SCANModel(pl.LightningModule):

    def __init__(
        self,
        model: SimCLRModel.SimCLRModel,
        n_classes: int = 10,
        entropy_lambda: float = 5.,
        lr: float = 1e-4,
        momentum: float = 0.9,
        epochs_max: int = 500,
        b1: float = 0.9,
        b2: float = 0.999,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        #
        # Access params via:
        # self.hparams.latent_dim
        self.save_hyperparameters('n_classes', 'lr', 'b1', 'b2', 'weight_decay', 'entropy_lambda')
        self.model = SCANModelPT(model.model, n_classes=n_classes)
        self.training_step_impl = self.SCAN_training_step
        self.epoch = 0

    def forward(self, x, training):
        if training:
            bs = x.shape[0]
            return self.model.resnet_feat(x).view(bs, -1)
        else:
            return self.model(x)

    def setSelfLabel(self):
        self.training_step_impl = self.self_label_training_step
        self.model.setSelfLabel()

    def self_label_training_step(self, batch, batch_idx, ret_loss_components=False):
        imgs, labels, idx, _, imgs_tf, labels_neighbor, idx_neighbor, _ = batch
        #
        # All images are strongly augmented like said in the paper.
        imgs = torch.cat([imgs, imgs_tf], 0)
        probs = self.model(imgs)
        probs, probs_tf = torch.split(probs, probs.shape[0] // 2, 0)
        loss = self.model.selfLabelLoss(probs, probs_tf)
        self.log('self_label_loss', loss)
        output = OrderedDict({
            'loss': loss,
        })
        return output

    def SCAN_training_step(self, batch, batch_idx, ret_loss_components=False):
        imgs, labels, idx, nearest, imgs_neighbor, labels_neighbor, idx_neighbor, neighbors_neighbors = batch
        imgs = torch.cat([imgs, imgs_neighbor], 0)
        probs = self.model(imgs)
        probs, probs_neighbor = torch.split(probs, probs.shape[0] // 2, 0)
        dotloss, entropy = self.model.SCANLoss(probs, probs_neighbor, idx, idx_neighbor, nearest, self.hparams.entropy_lambda)
        loss = dotloss - self.hparams.entropy_lambda * entropy
        #
        # Logging.
        self.log('loss', loss)
        self.log('dotloss', dotloss)
        self.log('entropy', entropy)
        output = OrderedDict({
            'loss': loss,
        })
        if ret_loss_components:
            output['dotloss'] = dotloss
            output['entropy'] = entropy
        return output

    def training_step(self, batch, batch_idx, ret_loss_components=False):
        return self.training_step_impl(batch, batch_idx, ret_loss_components)

    def validation_step(self, batch, batch_idx):
        out = self.training_step(batch, batch_idx, ret_loss_components=True)
        ret = {
                'val_loss': out['loss'],
                'loss': out['loss'],
                # 'log': {'val_loss': out['loss']}
            }
        if 'dotloss' in out:
            self.log('val_loss_unweighted', out['dotloss'] - out['entropy'])
            ret = {'val_loss': out['dotloss'] - out['entropy']}
        self.log('val_loss', ret['val_loss'])
        return ret

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)
        self.epoch += 1
        return {
            'avg_val_loss': avg_val_loss,
            'loss': avg_val_loss,
            'epoch': self.epoch - 1
        }

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        wd = self.hparams.weight_decay
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        return opt

    def on_epoch_end(self):
        pass
