import numpy as np
import torch
import torch.nn as nn
from . import debug as db
import torch.nn.functional as F
from collections import OrderedDict

from torchvision import models
import pytorch_lightning as pl

class SIMCLRModel(nn.Module):

    def __init__(
        self,
        insize: int = 512,
        hiddenSize: int = 128,
        cosine_temp: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.resnet_feat = torch.nn.Sequential(*list(models.resnet18().children())[:-1])
        self.cosine_temp = cosine_temp
        self.nn = nn.Sequential(*[
            nn.Linear(insize, hiddenSize),
            nn.ReLU(inplace=True),
            nn.Linear(hiddenSize, hiddenSize),
        ])

    def forward_impl(self, x: torch.tensor):
        bs = x.shape[0]
        return self.nn(self.resnet_feat(x).view(bs, -1))

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
        cosinesim -= cosinesim.max(-1)[0]
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

class SCANModel(pl.LightningModule):

    def __init__(
        self,
        lr: float = 1e-4,
        b1: float = 0.5,
        b2: float = 0.999,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        super().__init__()
        #
        # Access params via:
        # self.hparams.latent_dim
        self.save_hyperparameters()
        self.model = SIMCLRModel()

    def forward(self, x):
        return self.model(x)

    # def simCLRL(self, y_hat, y):
    #     return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_idx):
        imgs1, imgs2 = batch
        imgs1, _ = imgs1
        imgs2, _ = imgs2
        # z1 = self.model(imgs1)
        # z2 = self.model(imgs2)
        imgs = torch.cat([imgs1, imgs2])
        z = self.model(imgs)
        loss = self.model.simclr_loss(z)
        tqdm_dict = {'simclr_loss': loss}
        self.log('simclr_loss', loss)
        output = OrderedDict({
            'loss': loss,
            # 'progress_bar': tqdm_dict,
            # 'log': tqdm_dict
        })
        return output
        # # train generator
        # if optimizer_idx == 0:

        #     # # generate images
        #     # self.generated_imgs = self(z)

        #     # # log sampled images
        #     # sample_imgs = self.generated_imgs[:6]
        #     # grid = torchvision.utils.make_grid(sample_imgs)
        #     # self.logger.experiment.add_image('generated_images', grid, 0)

        #     # # ground truth result (ie: all fake)
        #     # # put on GPU because we created this tensor inside training_loop
        #     # valid = torch.ones(imgs.size(0), 1)
        #     # valid = valid.type_as(imgs)

        #     # # adversarial loss is binary cross-entropy
        #     # g_loss = self.adversarial_loss(self.discriminator(self(z)), valid)
        #     # tqdm_dict = {'g_loss': g_loss}
        #     # output = OrderedDict({
        #     #     'loss': g_loss,
        #     #     'progress_bar': tqdm_dict,
        #     #     'log': tqdm_dict
        #     # })
        #     return output

        # # train discriminator
        # if optimizer_idx == 1:
        #     # # Measure discriminator's ability to classify real from generated samples

        #     # # how well can it label as real?
        #     # valid = torch.ones(imgs.size(0), 1)
        #     # valid = valid.type_as(imgs)

        #     # real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

        #     # # how well can it label as fake?
        #     # fake = torch.zeros(imgs.size(0), 1)
        #     # fake = fake.type_as(imgs)

        #     # fake_loss = self.adversarial_loss(
        #     #     self.discriminator(self(z).detach()), fake)

        #     # # discriminator loss is the average of these
        #     # d_loss = (real_loss + fake_loss) / 2
        #     # tqdm_dict = {'d_loss': d_loss}
        #     # output = OrderedDict({
        #     #     'loss': d_loss,
        #     #     'progress_bar': tqdm_dict,
        #     #     'log': tqdm_dict
        #     # })
        #     return output

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2
        wd = self.hparams.weight_decay

        opt_m = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(b1, b2), weight_decay=wd)
        return opt_m

    def on_epoch_end(self):
        pass
        # z = self.validation_z.type_as(self.generator.model[0].weight)

        # # log sampled images
        # sample_imgs = self(z)
        # grid = torchvision.utils.make_grid(sample_imgs)
        # self.logger.experiment.add_image('generated_images', grid, self.current_epoch)
