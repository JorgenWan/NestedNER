import torch

from . import CRITERION_REGISTRY

def build_criterion(cfg):
    crite_name = cfg.criterion
    crite = CRITERION_REGISTRY[crite_name].build_criterion(cfg)

    return crite

def calculate_KLD_for_gaussian(u1, logv1, u2, logv2, reduce_mean=True):
    """
    KL Divergence of two gaussian p1 and p2,
    i.e. KL(p1||p2)
    """
    v1 = logv1.mul(0.5).exp()
    v2 = logv2.mul(0.5).exp()

    var_ratio = (v1 / v2).pow(2)
    t1 = ((u1 - u2) / v2).pow(2)
    kld = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

    if reduce_mean:
        kld = kld.mean() # todo: is mean right way? or we should sum?

    return kld

def torch_calculate_KLD_for_gaussian(u1, logv1, u2, logv2):
    """
    KL Divergence of two gaussian p1 and p2,
    i.e. KL(p1||p2)
    """
    p = torch.distributions.Normal(u1, logv1.mul(0.5).exp())
    q = torch.distributions.Normal(u2, logv2.mul(0.5).exp())
    kld = torch.distributions.kl_divergence(p, q).mean()

    return kld

if __name__ == "__main__":
    u1 = torch.FloatTensor([1,2])
    logv1 = torch.FloatTensor([1,1])
    u2 = torch.FloatTensor([0,0])
    logv2 = torch.FloatTensor([3,3])

    print(calculate_KLD_for_gaussian(u1, logv1, u2, logv2))
    print(old_calculate_KLD_for_gaussian(u1, logv1, u2, logv2))