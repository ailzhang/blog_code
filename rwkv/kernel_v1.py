import taichi as ti

Tmax = 768
fwd_block_dim=768
@ti.kernel
def timex_taichi_forward(out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    ti.loop_config(block_dim=fwd_block_dim) 
    for b, c, t in out:
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        k_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        w_pad[t] = w[c, t]
        k_pad[t] = k[b, c, t]
        ti.simt.block.sync()
        s = eps
        for u in range(0, t+1):
            s += w_pad[(T-1)-(t-u)] * k_pad[u]
        ti.simt.block.sync()
        out[b, c, t] = s

bwd_block_dim = 768
@ti.kernel
def timex_taichi_backward(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    ti.loop_config(block_dim=fwd_block_dim)
    for b, c, t in gwk:
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        k_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        g_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        w_pad[t] = w[c, t]
        k_pad[t] = k[b, c, t]
        g_pad[t] = gwk[b, c, t]
        ti.simt.block.sync()

        s = 0.0
        for u in range(0, t+1):
            s += g_pad[(T-1)-(t-u)] * k_pad[u]
        ti.simt.block.sync()
        gw[b, c, t] = s
        s = 0.0
        for u in range(t, T):
            s += g_pad[(T-1)+(t-u)] * w_pad[u]
        ti.simt.block.sync()
        gk[b, c, t] = s