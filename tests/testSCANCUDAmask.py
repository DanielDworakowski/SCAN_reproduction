import torch
import scancuda
from scanrepro import debug as db

if __name__ == '__main__':
    p_idx = torch.arange(256).cuda()
    nearest_neighors = torch.arange(10) * 1000
    nearest_neighors = nearest_neighors.repeat(256, 1).cuda()
    maskout = torch.zeros((p_idx.shape[0], p_idx.shape[0]), device=p_idx.device, dtype=torch.int32)
    db.printInfo(nearest_neighors)
    db.printTensor(nearest_neighors)
    db.printTensor(p_idx)
    db.printTensor(maskout)
    scancuda.SCAN_NN_Mask_Fill(p_idx, nearest_neighors, maskout)
    db.printInfo(maskout)
    db.printInfo(maskout[9])
    db.printInfo(nearest_neighors[0])
    # db.printInfo(maskout[9, nearest_neighors[0]])
    db.printTensor(maskout)
