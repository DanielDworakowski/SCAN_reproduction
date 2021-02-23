import torch
import tqdm
import random
import torchvision
from torchvision.datasets.utils import download_and_extract_archive
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from .RandAugment import RandAugment, CutoutStandAlone
from .SimCLRModel import SimCLRModel
from scanrepro import debug as db
import pytorch_lightning as pl

transformSettings = {
    True:
        transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4,0.4,0.4,0.1)],
                p=0.5),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]), # Simclr training.
    False:
        transforms.Compose([
            RandAugment(4, 9),
            CutoutStandAlone(0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ]) # RandAugment training.
}

DatasetDict = {
    'CIFAR10': torchvision.datasets.CIFAR10
}
class SIMCLRDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset, returnIDX):
        self.dataset = dataset
        self.getitemimpl = self.getItemNoIDX
        self.setReturnIDX(returnIDX)

    def __len__(self):
        return len(self.dataset)

    def setReturnIDX(self, returnIDX):
        if returnIDX:
            self.getitemimpl = self.getItemWithIDX

    def getItemNoIDX(self, idx):
        aug1 = self.dataset[idx]
        aug2 = self.dataset[idx]
        return (aug1, aug2)

    def getItemWithIDX(self, idx):
        aug1 = self.dataset[idx]
        aug2 = self.dataset[idx]
        return (aug1, aug2, idx)

    def __getitem__(self, idx):
        return self.getitemimpl(idx)

class SCANDatasetWrapper(torch.utils.data.Dataset):

    def __init__(self, dataset: torch.utils.data.Dataset, nearest_neighbours: str = 'nn.tgz'):
        self.dataset = dataset
        self.nearest_neighbours = torch.load(nearest_neighbours)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        nearest_neighbours = self.nearest_neighbours[idx]
        # nn_idx = torch.random.choice(nearest_neighbours[1:])
        nn_idx = nearest_neighbours[random.randint(1, len(nearest_neighbours)-1)]
        img_neighbor, target_neighbour = self.dataset[nn_idx]
        nearest_neighbours_neightbours = self.nearest_neighbours[nn_idx]
        return (img, target, idx, nearest_neighbours, img_neighbor, target_neighbour, nn_idx, nearest_neighbours_neightbours)
class GenericDataLoader(pl.LightningDataModule):

    def __init__(self,
                data_dir: str = './',
                batch_size: int = 512,
                num_workers: int = 8,
                dataset_name: str = 'CIFAR10',
                simclr: bool = False,
                train_fraction: float = 1.):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_fraction = train_fraction
        self.simclr = simclr
        #
        # Transformations according to the SCAN paper.
        self.transform = transformSettings[simclr]

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
            dset_full = self.dset(self.data_dir, train=True, transform=self.transform, download=True)
            self.dims = tuple(dset_full[0][0].shape)
            if self.simclr:
                dset_full = SIMCLRDatasetWrapper(dset_full, returnIDX)
            else:
                dset_full = SCANDatasetWrapper(dset_full)
            self.train = dset_full
            self.val = dset_full
            if self.train_fraction < 1.:
                train_length = int(self.train_fraction * len(dset_full))
                val_length = len(dset_full) - train_length
                self.train, self.val = random_split(dset_full, [train_length, val_length])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.test = self.dset(self.data_dir, train=False, transform=self.transform, download=True)
            self.dims = tuple(self.test[0][0].shape)
            self.test = SIMCLRDatasetWrapper(self.test, returnIDX)

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
    dm = GenericDataLoader(dataset_name=dataset_name, train_fraction=1., simclr=True)
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
