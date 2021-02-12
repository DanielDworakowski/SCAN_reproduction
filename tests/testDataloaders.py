import torch
import torchvision
from scanrepro import debug as db
from scanrepro import Dataloaders
from matplotlib import pyplot as plt

if __name__ == '__main__':
    dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', batch_size=16, num_workers=0)
    dm.setup()
    dl = dm.train_dataloader()
    for batch in dl:
        batch1, batch2 = batch
        img1,_ = batch1
        img2,_ = batch2
        #
        # Denormalize the data.
        img1 *= 0.5
        img1 += 0.5
        img2 *= 0.5
        img2 += 0.5
        db.printInfo(img1.max())
        db.printInfo(img1.min())
        grid = torchvision.utils.make_grid(img1, nrow=4)
        plt.imshow(grid.permute(1,2,0))
        plt.figure()
        grid = torchvision.utils.make_grid(img2, nrow=4)
        plt.imshow(grid.permute(1,2,0))
        plt.show()
