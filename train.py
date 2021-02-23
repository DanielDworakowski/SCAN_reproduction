#!/usr/bin/env python
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from scanrepro import Dataloaders, ScanModel, SimCLRModel, debug as db
torch.backends.cudnn.benchmark=True

def train_simclr(args):
    import pathlib
    model = SimCLRModel.SimCLRModel()
    if args.load_SCAN_checkpoint != '':
        return model
    elif args.load_simclr_checkpoint != '' and pathlib.Path(args.load_simclr_checkpoint).exists():
        db.printInfo('Loaded model from checkpoint')
        model = model.load_from_checkpoint(checkpoint_path=args.load_simclr_checkpoint)
    else:
        dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', simclr=True)
        checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor='loss',
                                                    dirpath='./checkpoints/',
                                                    filename='simclr-{epoch:02d}-{val_loss:.2f}',
                                                    mode='min')
        trainer = pl.Trainer.from_argparse_args(args,
                                                max_epochs=1000,
                                                gpus=-1,
                                                sync_batchnorm=True,
                                                check_val_every_n_epoch=10,
                                                callbacks=[checkpoint_cb])
        trainer.fit(model, dm)
    return model

def train_SCAN(model: ScanModel.SCANModel, simclrmodel: SimCLRModel.SimCLRModel, args):
    import pathlib
    if args.load_SCAN_checkpoint != '' and pathlib.Path(args.load_SCAN_checkpoint).exists():
        db.printInfo('Loaded model from checkpoint')
        db.printInfo(args.load_SCAN_checkpoint)
        model = model.load_from_checkpoint(checkpoint_path=args.load_SCAN_checkpoint, model=simclrmodel)
    else:
        dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', simclr=False)
        checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor='loss',
                                                    dirpath='./checkpoints/',
                                                    filename='SCAN-{epoch:02d}-{val_loss:.2f}',
                                                    mode='min')
        trainer = pl.Trainer.from_argparse_args(args,
                                                gpus=-1,
                                                sync_batchnorm=True,
                                                check_val_every_n_epoch=10,
                                                callbacks=[checkpoint_cb])
        trainer.fit(model, dm)
    return model

def train_self_label(model: ScanModel.SCANModel, args):
    model.setSelfLabel()
    dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', simclr=False)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(monitor='loss',
                                                dirpath='./checkpoints/',
                                                filename='SCAN_self_label-{epoch:02d}-{val_loss:.2f}',
                                                mode='min')
    trainer = pl.Trainer.from_argparse_args(args,
                                            gpus=-1,
                                            sync_batchnorm=True,
                                            check_val_every_n_epoch=10,
                                            callbacks=[checkpoint_cb])
    trainer.fit(model, dm)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--load_simclr_checkpoint', type=str, default='')
    parser.add_argument('--load_SCAN_checkpoint', type=str, default='')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--calculateNN', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    simclr_model = train_simclr(args).cuda()


    # ------------
    # data
    # ------------

    # ------------
    # model
    # ------------
    model = ScanModel.SCANModel(simclr_model)
    Dataloaders.get_knn_dataset(simclr_model, 'CIFAR10', args.knn, n_image_samples=10, force_load=args.calculateNN)
    model = train_SCAN(model, simclr_model, args)

    train_self_label(model, args)
    # ------------
    # training
    # ------------


    # ------------
    # testing
    # ------------
    result = trainer.test(dm)
    print(result)


if __name__ == '__main__':
    cli_main()
