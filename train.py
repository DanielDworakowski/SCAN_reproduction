from argparse import ArgumentParser
import pytorch_lightning as pl
from scanrepro import Dataloaders, ScanModel


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    # parser.add_argument('--sync_batchnorm', type=bool, default=True)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10')

    # ------------
    # model
    # ------------
    model = ScanModel.SCANModel()

    # ------------
    # training
    # ------------
    checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor='simclr_loss',
                                                dirpath='./checkpoints/',
                                                filename='simclr-{epoch:02d}-{val_loss:.2f}',
                                                mode='min')
    trainer = pl.Trainer.from_argparse_args(args,
                                            sync_batchnorm=True,
                                            callbacks=[checkpoint_cb])
    trainer.fit(model, dm)

    # ------------
    # testing
    # ------------
    result = trainer.test(dm)
    print(result)


if __name__ == '__main__':
    cli_main()
