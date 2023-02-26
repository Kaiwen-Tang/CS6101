import torch
from colossalai.tensor import ComputePattern, ComputeSpec, DistSpecManager, ProcessGroup, ShardSpec

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def init_1d_row_for_linear_weight_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([-1], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if 'weight' in n and 'norm' not in n and 'patch_embed.proj.weight' not in n:
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)

def init_1d_col_for_linear_weight_bias_spec(model, world_size: int):
    pg = ProcessGroup(tp_degree=world_size)
    spec = (ShardSpec([0], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        for n, p in model.named_parameters():
            if ('weight' in n or 'bias' in n) and 'norm' not in n and ('patch_embed.proj.weight' not in n
                                                                       and 'patch_embed.proj.bias' not in n):
                p.set_process_group(pg)
                p.set_tensor_spec(*spec)


def init_spec_func(model, tp_type):
    world_size = torch.distributed.get_world_size()
    if tp_type == 'row':
        init_1d_row_for_linear_weight_spec(model, world_size)
    elif tp_type == 'col':
        init_1d_col_for_linear_weight_bias_spec(model, world_size)
    else:
        raise NotImplemented