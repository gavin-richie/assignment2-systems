import torch
import torch.distributed as dist

class DDPOverlap(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()

        rank = 0
        self.handles = []
        self.module = module

        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(tensor=param.data, src=rank)

        # use hook to enable communication in between grad accumulations!
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.add_hook(param))

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data /= self.world_size

        self.handles.clear()

    def add_hook(self, param):
        def hook(param):
            handle = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)
            # we want to enable async now, so true!
            # avg not allwoed here, divide by n workers later

        return hook
