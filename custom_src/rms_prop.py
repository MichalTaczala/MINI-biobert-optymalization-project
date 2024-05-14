# import torch


# class RMSProp:
#     def __init__(self, parameters, lr=0.001, alpha=0.9, eps=1e-8):
#         self.parameters = list(parameters)
#         self.lr = lr
#         self.alpha = alpha
#         self.eps = eps
#         self.sq_grads = [torch.zeros_like(p) for p in self.parameters]

#     def step(self):
#         for p, sq_grad in zip(self.parameters, self.sq_grads):
#             if p.grad is not None:
#                 grad = p.grad.data
#                 sq_grad.mul_(self.alpha).addcmul_(1 - self.alpha, grad, grad)
#                 adjusted_lr = self.lr / (torch.sqrt(sq_grad) + self.eps)
#                 p.data -= adjusted_lr * grad

import torch
from torch.optim.optimizer import Optimizer, required


class HybridSGDRMSprop(torch.optim.Optimizer):
    def __init__(
        self, params, lr=0.001, alpha=0.9, eps=1e-8, weight_decay=0, momentum=0
    ):
        defaults = dict(
            lr=lr, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum
        )
        super(HybridSGDRMSprop, self).__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["square_avg"] = torch.zeros_like(p.data)
                        if group["momentum"] > 0:
                            state["momentum_buffer"] = torch.zeros_like(p.data)

                    square_avg = state["square_avg"]
                    alpha = group["alpha"]
                    state["step"] += 1

                    if group["weight_decay"] != 0:
                        grad.add_(p.data, alpha=group["weight_decay"])

                    square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                    if group["momentum"] > 0:
                        buf = state["momentum_buffer"]
                        buf.mul_(group["momentum"]).addcdiv_(
                            grad, square_avg.sqrt().add(group["eps"])
                        )
                        p.data.add_(-group["lr"], buf)
                    else:
                        p.data.addcdiv_(
                            -group["lr"], grad, square_avg.sqrt().add(group["eps"])
                        )
