import torch
from scanrepro import debug as db
from scanrepro.SimCLRModel import SimCLRModel
from torch import nn

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature


    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask
        db.printInfo(mask)

        # Log-softmax
        db.printInfo(logits)
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss


if __name__ == '__main__':
    mdl = SimCLRModel()
    inp1 = torch.arange(9).view(3,3).float()
    inp2 = torch.arange(9).view(3,3).float()
    z_sim = torch.cat([inp1, inp2])

    sim = mdl.model.cosine_similarity(z_sim, z_sim)
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
    db.printTensor(z_sim)
    db.printInfo(mdl.model.simclr_loss(z_sim))
    db.printInfo(inp1)
    db.printInfo(torch.norm(inp1, dim=1))

    in1 = inp1 / torch.norm(inp1, dim=1).view(-1,1)
    in2 = inp2 / torch.norm(inp2, dim=1).view(-1,1)

    in1 = in1.view(3, -1, 3)
    in2 = in2.view(3, -1, 3)

    # in1 = inp1.view(3, -1, 3) / torch.norm(inp1, dim=0)
    # in2 = inp2.view(3, -1, 3) / torch.norm(inp2, dim=0)
    db.printInfo(in1)
    db.printInfo(in2)
    new_z = torch.cat([in1, in2], 1)
    l = SimCLRLoss(0.1)
    db.printInfo(l(new_z.cuda()))
