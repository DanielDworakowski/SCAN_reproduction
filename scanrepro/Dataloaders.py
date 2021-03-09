import torch
import tqdm
import random
import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from .RandAugment import CutoutDefault, RandAugment
from .SimCLRModel import SimCLRModel
from scanrepro import debug as db
import pytorch_lightning as pl

transformSettings = {
    'simclr':
        transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4,0.4,0.4,0.1)],
                p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]), # Simclr training.
    'SCAN':
        # transforms.Compose([
        #     transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply(
        #         [transforms.ColorJitter(0.4,0.4,0.4,0.1)],
        #         p=0.8),
        #     transforms.RandomGrayscale(p=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # ]), # Simclr training.
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32),
            # transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            RandAugment(4),
            # CutoutStandAlone(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            CutoutDefault(1, 16),
        ]), # RandAugment training.
    'selflabel':
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32),
            RandAugment(4),
            # CutoutStandAlone(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
            CutoutDefault(1, 16),
        ]), # RandAugment training.
    'default':
        transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        ])
}

DatasetDict = {
    'CIFAR10': torchvision.datasets.CIFAR10
}
class SIMCLRDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, returnIDX, transform=None):
        self.dataset = dataset
        self.getitemimpl = self.getItemNoIDX
        self.setReturnIDX(returnIDX)
        self.transform = transform
        assert transform is not None

    def __len__(self):
        return len(self.dataset)

    def setReturnIDX(self, returnIDX):
        if returnIDX:
            self.getitemimpl = self.getItemWithIDX

    def getItemNoIDX(self, idx):
        img1, data1 = self.dataset[idx]
        aug1 = self.transform(img1)
        img2, data2 = self.dataset[idx]
        aug2 = self.transform(img2)
        return ((aug1, data1), (aug2, data2))

    def getItemWithIDX(self, idx):
        img1, data1 = self.dataset[idx]
        aug1 = self.transform(img1)
        img2, data2 = self.dataset[idx]
        aug2 = self.transform(img2)
        return ((aug1, data1), (aug2, data2), idx)

    def __getitem__(self, idx):
        return self.getitemimpl(idx)

class SCANDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, nearest_neighbours: str = 'nn.tgz', transform=None):
        self.dataset = dataset
        self.nearest_neighbours = torch.load(nearest_neighbours)
        self.transform = transform
        assert transform is not None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        nearest_neighbours = self.nearest_neighbours[idx]
        # nn_idx = torch.random.choice(nearest_neighbours[1:])
        nn_idx = nearest_neighbours[random.randint(1, len(nearest_neighbours)-1)]
        img_neighbor, target_neighbour = self.dataset[nn_idx]
        nearest_neighbours_neightbours = self.nearest_neighbours[nn_idx]
        return (self.transform(img), target, idx, nearest_neighbours, self.transform(img_neighbor), target_neighbour, nn_idx, nearest_neighbours_neightbours)

class SelfTransformedWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.idTransform = transformSettings['default']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        img = self.idTransform(img)
        img_aug, target_aug = self.dataset[idx]
        img_aug = self.transform(img_aug)
        return (img, target, idx, torch.tensor([]), img_aug, target_aug, idx, torch.tensor([]))

class GenericDataLoader(pl.LightningDataModule):

    def __init__(self,
                data_dir: str = './',
                batch_size: int = 512,
                num_workers: int = 8,
                dataset_name: str = 'CIFAR10',
                mode: str = 'simclr',
                train_fraction: float = 1.):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.mode = mode
        #
        # Transformations according to the SCAN paper.
        self.transform = transformSettings[mode]

        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        # ])

        self.dset = DatasetDict[dataset_name]
        if dataset_name not in DatasetDict:
            raise ValueError('Dataset provided not in dataset dict')

    def prepare_data(self):
        pass
        # self.dset(self.data_dir, train=True, download=True)
        # self.dset(self.data_dir, train=False, download=True)

    def setup(self, stage=None, returnIDX=False):
        if self.dset is None:
            raise RuntimeError('Must specify a dataset.')
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            # dset_full = self.dset(self.data_dir, train=True, transform=self.transform, download=True)
            dset_full = self.dset(self.data_dir, train=True, download=True)
            if self.mode == 'simclr':
                self.train = SIMCLRDatasetWrapper(dset_full, returnIDX, transform=self.transform)
                self.val = SIMCLRDatasetWrapper(dset_full, returnIDX, transform=transformSettings['default'])
            elif self.mode == 'SCAN':
                self.train = SCANDatasetWrapper(dset_full, transform=self.transform)
                self.val = SCANDatasetWrapper(dset_full, transform=transformSettings['default'])
            elif self.mode == 'selflabel':
                self.train = SelfTransformedWrapper(dset_full, transform=self.transform)
                self.val = SelfTransformedWrapper(dset_full, transform=transformSettings['default'])

            self.dims = tuple(self.train[0][0][0].shape)
            # self.train = dset_full
            # self.val = dset_full
            if self.train_fraction < 1.:
                train_length = int(self.train_fraction * len(dset_full))
                val_length = len(dset_full) - train_length
                self.train, self.val = random_split(dset_full, [train_length, val_length])

        # Assign test dataset for use in dataloader(s)
        # if stage == 'test' or stage is None:
        #     self.test = self.dset(self.data_dir, train=False, transform=self.transform, download=True)
        #     self.dims = tuple(self.test[0][0].shape)
        #     self.test = SIMCLRDatasetWrapper(self.test, returnIDX)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

@torch.no_grad()
def get_knn_dataset(simclr_model: SimCLRModel,
                    dataset_name: str,
                    knn: int,
                    n_image_samples: int = 2,
                    save_name: str = 'nn.tgz',
                    force_load: bool = False):
    import pathlib
    if pathlib.Path(save_name).exists() and not force_load:
        return
    dm = GenericDataLoader(dataset_name=dataset_name, train_fraction=1., mode='simclr')
    dm.setup(returnIDX=True)
    dataset = dm.train_dataloader()
    #
    # What to do with the generated augmentations while this is being generated?
    # They had a very big increase in performance after false postives were removed from their set of KNN.
    codes = torch.zeros(len(dm.train), simclr_model.hparams.in_size, device=simclr_model.device)
    db.printInfo(n_image_samples)
    for i in range(max(n_image_samples // 2, 1)):
        for batch in tqdm.tqdm(dataset, 'Image Code Generation %d/%d: ' % (i, n_image_samples//2)):
            imgs1, imgs2, idx = batch
            imgs1 = imgs1[0].to(simclr_model.device)
            imgs2 = imgs2[0].to(simclr_model.device)
            idx = idx.to(simclr_model.device)
            code = simclr_model(imgs1, feature_extraction=True)
            code2 = simclr_model(imgs2, feature_extraction=True)
            codes[idx] += code / n_image_samples
            codes[idx] += code2 / n_image_samples
    #
    # Now iterating over the dataset find the top-k closest samples for each image index.
    closest_idxes = []
    for img_idx in tqdm.tqdm(range(len(dm.train)), 'top-k image generation: '):
        code = codes[img_idx:img_idx+1]
        distances = (codes - code).norm(dim=-1)
        #
        # Take knn+1 neightbours since it is guarunteed to be closest to itself.
        _, closest_idx = distances.topk(k=knn+1, largest=False, sorted=True)
        closest_idxes.append(closest_idx.cpu().unsqueeze(0))
    closest_idxes = torch.cat(closest_idxes, 0)
    torch.save(closest_idxes, save_name)
