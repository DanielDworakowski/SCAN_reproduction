from scanrepro import Dataloaders
import tqdm

if __name__ == '__main__':
    dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', batch_size=256, num_workers=16, simclr=False, train_fraction=1.)
    dm.setup()
    # train_loader = dm.train_dataloader()
    dataset = dm.train

    total_cnt = 0
    n_correct = 0
    pbar = tqdm.tqdm(dataset)
    for batch in pbar:
        img, label, idx, nearest, _, _, _, _ = batch
        for nearest_idx in nearest:
            _, label_neighor, _, _, _, _, _, _ = dataset[nearest_idx]
            n_correct += (label_neighor == label)
            total_cnt += 1
        pbar.set_description('Acc: %0.3f' % (n_correct / total_cnt))
