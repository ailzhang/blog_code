import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
torch.backends.cudnn.benchmark = True
# turn off TF32 for higher accuracy
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

import time
import taichi as ti

# Change this to use different optimized versions
from kernel_v3 import timex_taichi_forward, timex_taichi_backward

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

def RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps):
    # this is the formula (very slow)
    out = torch.empty((B, C, T), device='cuda')
    for b in range(B):
        for c in range(C):
            for t in range(T):
                s = eps
                for u in range(0, t+1):
                    s += w[c][(T-1)-(t-u)] * k[b][c][u]
                out[b][c][t] = s
    return out


def RUN_PYTORCH(w, k, B, C, T, eps):
    # this shall equal the formula
    return F.conv1d(nn.ZeroPad2d((T-1, 0, 0, 0))(k), w.unsqueeze(1), groups=C) + eps

class TimeX_Taichi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, k, B, C, T, eps):
        ctx.B = B
        ctx.C = C
        ctx.T = T
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w = w.contiguous()
        k = k.contiguous()
        ctx.save_for_backward(w, k)
        wk = torch.empty((B, C, T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_forward(wk, w, k, B, C, T, eps)
        ti.sync()
        return wk

    @staticmethod
    def backward(ctx, gwk):
        #assert ctx.T % 4 == 0 and ctx.T <= T_MAX and ctx.B % B_GROUP_FORWARD == 0 and ctx.B % B_GROUP_BACKWARD == 0, "require T % 4 == 0 and T <= T_MAX and B % B_GROUP_* == 0"
        w, k = ctx.saved_tensors
        gw = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        gk = torch.empty((ctx.B, ctx.C, ctx.T), device='cuda',
                         memory_format=torch.contiguous_format)
        timex_taichi_backward(w, k, gwk.contiguous(), gw,
                            gk, ctx.B, ctx.C, ctx.T)
        ti.sync()
        # actually pytorch will do gw.sum(dim=0) but we will do it anyway just to be safe
        return (gw.sum(dim=0), gk, None, None, None, None)

def RUN_TAICHI(w, k, B, C, T, eps):
    return TimeX_Taichi.apply(w.cuda(), k.cuda(), B, C, T, eps)



######################################################################################################
# Check correctness & speed benchmark
######################################################################################################

def CHECK_PYTORCH():
    B = 3
    C = 5
    T = 11
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    r0 = RUN_FORMULA_VERY_SLOW(w, k, B, C, T, eps)
    r1 = RUN_PYTORCH(w, k, B, C, T, eps)

    print('--> pytorch correct =', torch.allclose(r0, r1),
          ', err ratio =', get_err_ratio(r0, r1))

def CHECK_TAICHI(silent=False):
    B = 32
    C = 768
    T = 768
    eps = 0.1

    set_seed(42)
    w = torch.rand(C, T, requires_grad=True, device='cuda')
    k = torch.rand(B, C, T, requires_grad=True, device='cuda')

    # check forward

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        r1 = RUN_PYTORCH(w, k, B, C, T, eps)
    if not silent:
        print('pytorch forward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))

    # check backward
    # a strange loss for better verification
    loss1 = ((r1 * r1) - torch.tanh(r1)).sum()
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        loss1.backward()
    if not silent:
        print('pytorch backward\n', prof.key_averages(group_by_stack_n=5).table(
            sort_by='self_cuda_time_total', row_limit=5))
    gw1 = w.grad.data.clone()
    gk1 = k.grad.data.clone()

    w.grad.data.zero_()
    k.grad.data.zero_()

    w.grad.data.zero_()
    k.grad.data.zero_()

    # Check Taichi
    ti.init(arch=ti.cuda, kernel_profiler=True)
    # Taichi
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    w.grad.data.zero_()
    k.grad.data.zero_()
    ti.profiler.clear_kernel_profiler_info()
    r3 = RUN_TAICHI(w, k, B, C, T, eps)
    ti.sync()

    print('--> Taichi fwd correct =', torch.allclose(r1, r3),
         ', err ratio =', get_err_ratio(r1, r3))
    loss3 = ((r3 * r3) - torch.tanh(r3)).sum()
    loss3.backward()
    if not silent:
        ti.profiler.print_kernel_profiler_info('trace')
    gw3 = w.grad.data.clone()
    gk3 = k.grad.data.clone()

    print('--> bwd gradW correct =', torch.allclose(gw1, gw3),
         ', err ratio =', get_err_ratio(gw1, gw3))
    print('--> bwd gradK correct =', torch.allclose(gk1, gk3),
         ', err ratio =', get_err_ratio(gk1, gk3))



if __name__ == "__main__":
    print('Taichi benchmark...')
    CHECK_TAICHI(silent=False)  # benchmark
