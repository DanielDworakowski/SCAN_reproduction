#!/usr/bin/env python
import tqdm
import torch
from argparse import ArgumentParser
import pytorch_lightning as pl
from scanrepro import Dataloaders, ScanModel, SimCLRModel, debug as db
import numpy as np
from scipy.optimize import linear_sum_assignment

torch.backends.cudnn.benchmark=True

#
# Taken from the original paper's work.
# https://github.com/wvangansbeke/Unsupervised-Classification/blob/master/utils/evaluate_utils.py
@torch.no_grad()
def hungarian_evaluate(targets, predictions, probs, class_names=None,
                        compute_purity=True, compute_confusion_matrix=True,
                        confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    # head = all_predictions[subhead_index]
    # targets = head['targets'].cuda()
    # predictions = head['predictions'].cuda()
    # probs = head['probabilities'].cuda()
    targets = targets.to(predictions.device)
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).to(predictions.device)
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)


    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)


    return {'ACC': acc, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


@torch.no_grad()
def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--load_SCAN_checkpoint', type=str, default='')
    parser.add_argument('--knn', type=int, default=10)
    parser.add_argument('--calculateNN', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    model = ScanModel.SCANModel(SimCLRModel.SimCLRModel())
    model = model.load_from_checkpoint(checkpoint_path=args.load_SCAN_checkpoint, model=SimCLRModel.SimCLRModel())
    model = model.cuda()
    dm = Dataloaders.GenericDataLoader(dataset_name='CIFAR10', mode='selflabel')
    dm.setup()
    probs = []
    targets = []
    # batchidx = 0
    for batch in tqdm.tqdm(dm.val_dataloader()):
        imgs, labels, idx, nearest, imgs_neighbor, labels_neighbor, idx_neighbor, neighbors_neighbors = batch
        targets.append(labels)
        imgs = imgs.cuda()

        probs.append(model(imgs, training=False))
        # batchidx+=1
        # if batchidx % 3 == 0:
        #     break
    probs = torch.cat(probs, 0)
    vals, preds = probs.max(dim=1)
    targets = torch.cat(targets, 0)
    score = hungarian_evaluate(targets, preds, probs)
    print(score)
    # ------------
    # testing
    # ------------


if __name__ == '__main__':
    cli_main()
