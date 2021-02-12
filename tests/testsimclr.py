import torch
from scanrepro import debug as db
from scanrepro.ScanModel import SIMCLRModel

if __name__ == '__main__':
    mdl = SIMCLRModel()
    inp1 = torch.arange(9).view(3,3).float()
    inp2 = torch.arange(9).view(3,3).float()
    z_sim = torch.cat([inp1, inp2])

    sim = mdl.cosine_similarity(z_sim, z_sim)
    sol = torch.tensor([[1.0000, 0.8854, 0.8427, 1.0000, 0.8854, 0.8427],
        [0.8854, 1.0000, 0.9964, 0.8854, 1.0000, 0.9964],
        [0.8427, 0.9964, 1.0000, 0.8427, 0.9964, 1.0000],
        [1.0000, 0.8854, 0.8427, 1.0000, 0.8854, 0.8427],
        [0.8854, 1.0000, 0.9964, 0.8854, 1.0000, 0.9964],
        [0.8427, 0.9964, 1.0000, 0.8427, 0.9964, 1.0000]])
    test_val = torch.all(torch.abs(sim - sol) < 1e-3).item()
    assert test_val, 'Check cosine_sim'
    # inp1 = torch.arange(4).view(2,2).float()
    # inp2 = torch.arange(4).view(2,2).float()
    # z_sim = torch.cat([inp1, inp2])
    # db.printInfo(z_sim)
    db.printInfo(mdl.simclr_loss(z_sim))
