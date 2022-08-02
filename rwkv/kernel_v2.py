import taichi as ti 

fwd_t_group = 4
Tmax = 768
fwd_block_dim=192
ti_matf = ti.types.vector(fwd_t_group, dtype=float)
@ti.kernel
def timex_taichi_forward(
        out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    ti.loop_config(block_dim=fwd_block_dim)
    for b, c, t_block in ti.ndrange(B, C, T // fwd_t_group):
        # Group both b and t with factor 4
        t = t_block * fwd_t_group
        s_mat = ti_matf(((eps,) * fwd_t_group,))
        k_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for i in ti.static(range(fwd_t_group)):
            k_pad[t + i] = k[b, c, t + i]
            w_pad[t + i] = w[c, t + i]
        ti.simt.block.sync()
        for u in range(0, t+1):
            for i in ti.static(range(fwd_t_group)):
                s_mat[i] += w_pad[(T-1)-(t-(u-i))] * k_pad[u]
        ti.simt.block.sync()
        # Compute the remaining triangle in the thread group.
        for i in ti.static(range(1, fwd_t_group)):
            for j in ti.static(range(i)):
                s_mat[i] += w_pad[T - i + j] * k_pad[t+1+j]
        ti.simt.block.sync()
        for i in ti.static(range(fwd_t_group)):
            out[b, c, t+ i] = s_mat[i]

bwd_t_group = 4
bwd_block_dim = 192
ti_back_matf = ti.types.vector(bwd_t_group, dtype=float)
@ti.kernel
def timex_taichi_backward(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    ti.loop_config(block_dim=bwd_block_dim)
    for b, c, t_block in ti.ndrange(B, C, T // bwd_t_group):
        t = t_block * bwd_t_group
        s = ti_back_matf(((0.0,) * bwd_t_group,))
        gwk_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        k_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for i in ti.static(range(bwd_t_group)):
            k_pad[t + i] = k[b, c, t + i]
            w_pad[t + i] = w[c, t + i]
            gwk_pad[t + i] = gwk[b, c, t + i]
        ti.simt.block.sync()
        for u in range(0, t+1):
            for i in ti.static(range(0, bwd_t_group)):
                s[i] += gwk_pad[(T-1)-(t+i-u)] * k_pad[u]
        ti.simt.block.sync()
        # The last triangle is specialized
        # u is replaced with t+1+j
        for i in ti.static(range(1, bwd_t_group)):
            for j in ti.static(range(i)):
                s[i] += gwk_pad[T-i+j] * k_pad[t+1+j]
        ti.simt.block.sync()
        # write out
        for i in ti.static(range(bwd_t_group)):
            gw[b, c, t+i] = s[i]

        s = ti_back_matf(((0.0,) * bwd_t_group,))
        # The first triangle is specialized
        # t' = t + i
        # u' = t' + j
        for i in ti.static(range(0, bwd_t_group-1)):
            for j in ti.static(range(i, bwd_t_group-1)):
                s[i] += gwk_pad[T+i-j-1] * w_pad[t+j]
        ti.simt.block.sync()

        for u in range(t+bwd_t_group-1, T):
            for i in ti.static(range(0, bwd_t_group)):
                s[i] += gwk_pad[(T-1)+(t+i-u)] * w_pad[u]
        ti.simt.block.sync()
        # write out
        for i in ti.static(range(bwd_t_group)):
            gk[b, c, t+i] = s[i]
