import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from .RandAugment import RandAugment
import pytorch_lightning as pl

DatasetDict = {
    'CIFAR10': torchvision.datasets.CIFAR10
}

class SIMCLR_DatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        aug1 = self.dataset[idx]
        aug2 = self.dataset[idx]
        return (aug1, aug2)

class GenericDataLoader(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size: int = 128, num_workers: int = 8, dataset_name: str = 'CIFAR10', simclr: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            RandAugment(4, 9),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
        if dataset_name not in DatasetDict:
            raise ValueError('Dataset provided not in dataset dict')
        self.dset = DatasetDict[dataset_name]

    def prepare_data(self):
        pass
        # self.dset(self.data_dir, train=True, download=True)
        # self.dset(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if self.dset is None:
            raise RuntimeError('Must specify a dataset.')
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            dset_full = self.dset(self.data_dir, train=True, transform=self.transform)
            self.dims = tuple(dset_full[0][0].shape)
            dset_full = SIMCLR_DatasetWrapper(dset_full)
            train_length = int(0.9 * len(dset_full))
            val_length = len(dset_full) - train_length
            self.train, self.val = random_split(dset_full, [train_length, val_length])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = self.dset(self.data_dir, train=False, transform=self.transform)
            self.dims = tuple(self.test[0][0].shape)
            self.test = SIMCLR_DatasetWrapper(self.test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)
