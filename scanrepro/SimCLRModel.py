import numpy as np
import torch
import torch.nn as nn
from . import debug as db
import torch.nn.functional as F
from collections import OrderedDict

from torchvision import models
import pytorch_lightning as pl

class SIMCLRModelPT(nn.Module):

    def __init__(
        self,
        in_size: int = 512,
        out_size: int = 128,
        cosine_temp: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.resnet_feat = torch.nn.Sequential(*list(models.resnet18().children())[:-1])
        self.cosine_temp = cosine_temp
        self.nn = nn.Sequential(*[
            nn.Linear(in_size, in_size),
            nn.ReLU(inplace=True),
            nn.Linear(in_size, out_size),
        ])

    def forward_feat(self, x: torch.tensor):
        bs = x.shape[0]
        return self.resnet_feat(x).view(bs, -1)

    def forward_impl(self, x: torch.tensor):
        bs = x.shape[0]
        h = self.resnet_feat(x).view(bs, -1)
        # db.printInfo(self.simclr_loss(h))
        return self.nn(h)

    def forward(self, x):
        return self.forward_impl(x)

    def cosine_similarity(self, z1, z2):
        num = z1 @ z2.transpose(0,1)
        norm_z1 = z1.norm(dim=-1)
        norm_z2 = z2.norm(dim=-1).view(-1, 1)
        sim = num / (norm_z1 * norm_z2 + 1e-9)
        return sim

    def simclr_loss(self, z):
        num_samples = z.shape[0] // 2
        #
        # Cosine similarity between every example.
        cosinesim = self.cosine_similarity(z, z)
        # cosinesim.div_(self.cosine_temp)
        cosinesim /= self.cosine_temp
        #
        # Numerical stability -- softmax is invariant to shift.
        cosinesim -= cosinesim.max(-1)[0].detach()
        cosinesim = cosinesim.exp()
        #
        # Zero out all the diagonal elements.
        mask = 1.0 - torch.eye(*cosinesim.shape, device=cosinesim.device, dtype=cosinesim.dtype)
        cosinesim = cosinesim * mask
        #
        # Sum all samples at each row to get the denominator.
        denom = torch.log(cosinesim.sum(-1))
        #
        # The postive samples are the diagonals of the upper right and lower left quadrants.
        num = -torch.cat([cosinesim.diagonal(num_samples), cosinesim.diagonal(-num_samples)]).log_()
        #
        # The final loss is the mean of this sum.
        loss = torch.mean(num + denom)
        return loss

class SimCLRModel(pl.LightningModule):

    def __init__(
        self,
        lr: float = 0.4,
        momentum: float = 0.9,
        # b1: float = 0.5,
        # b2: float = 0.999,
        weight_decay: float = 1e-4,
        in_size: int = 512,
        max_epochs: int = 500,
        **kwargs
    ):
        super().__init__()
        #
        # Access params via:
        # self.hparams.latent_dim
        self.save_hyperparameters()
        self.model = SIMCLRModelPT(in_size=in_size)

    def forward(self, x, feature_extraction=False):
        if feature_extraction:
            bs = x.shape[0]
            return self.model.resnet_feat(x).view(bs, -1)
            # return self.model.foward_feat(x)
        else:
            return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs1, imgs2 = batch
        imgs1, _ = imgs1
        imgs2, _ = imgs2
        imgs = torch.cat([imgs1, imgs2])
        z = self.model(imgs)
        loss = self.model.simclr_loss(z)
        self.log('loss', loss)
        output = OrderedDict({
            'loss': loss,
        })
        return output

    def validation_step(self, batch, batch_idx):
        out = self.training_step(batch, batch_idx)
        self.log('val_loss', out['loss'])
        return {
            'val_loss': out['loss'],
            # 'log': {'val_loss': out['loss']}
        }

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_val_loss)
        return {
            'avg_val_loss': avg_val_loss,
            # 'log': {'avg_val_loss': avg_val_loss}
        }

    def configure_optimizers(self):
        lr = self.hparams.lr
        # b1 = self.hparams.b1
        # b2 = self.hparams.b2
        # wd = self.hparams.weight_decay
        opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=self.hparams.momentum)
        eta_min = lr * 0.1 ** 3
        epochs = self.hparams.max_epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=eta_min)
        # opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        return [opt], [scheduler]


    def on_epoch_end(self):
        pass
