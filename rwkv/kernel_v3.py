import taichi as ti 

fwd_t_group = 6
fwd_b_group = 8
Tmax = 768
fwd_block_dim=128
ti_matf = ti.types.matrix(fwd_b_group, fwd_t_group, dtype=float)
@ti.kernel
def timex_taichi_forward(
        out: ti.types.ndarray(field_dim=3),
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32, eps: ti.f32):
    ti.loop_config(block_dim=fwd_block_dim)
    for b_block, c, t_block in ti.ndrange(B // fwd_b_group, C, T // fwd_t_group):
        # Group both b and t with factor 4
        t = t_block * fwd_t_group
        b = b_block * fwd_b_group
        s_mat = ti_matf(((eps,) * fwd_t_group,) * fwd_b_group)
        k_pad = ti.simt.block.SharedArray((fwd_b_group, Tmax), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for bi in ti.static(range(fwd_b_group)):
            for i in ti.static(range(fwd_t_group)):
                k_pad[bi, t + i] = k[b + bi, c, t + i]
                if bi == 0:
                    w_pad[t + i] = w[c, t + i]
        ti.simt.block.sync()
        for u in range(0, t+1):
            for bi in ti.static(range(fwd_b_group)):
                k_val = k_pad[bi, u]
                for i in ti.static(range(fwd_t_group)):
                    s_mat[bi, i] += w_pad[(T-1)-(t-(u-i))] * k_val
        ti.simt.block.sync()
        # Compute the remaining triangle in the thread group.
        for bi in ti.static(range(fwd_b_group)):
            for i in ti.static(range(1, fwd_t_group)):
                for j in ti.static(range(i)):
                    s_mat[bi, i] += w_pad[T - i + j] * k_pad[bi, t+1+j]
            for i in ti.static(range(fwd_t_group)):
                out[b + bi, c, t+ i] = s_mat[bi, i]

bwd_t_group = 6
bwd_b_group = 4
bwd_block_dim = 128
ti_back_matf = ti.types.matrix(bwd_b_group, bwd_t_group, dtype=float)
@ti.kernel
def timex_taichi_backward(
        w: ti.types.ndarray(field_dim=2),
        k: ti.types.ndarray(field_dim=3),
        gwk: ti.types.ndarray(field_dim=3),
        gw: ti.types.ndarray(field_dim=3),
        gk: ti.types.ndarray(field_dim=3),
        B: ti.i32, C: ti.i32, T: ti.i32):
    ti.loop_config(block_dim=bwd_block_dim)
    for b_block, c, t_block in ti.ndrange(B // bwd_b_group, C, T // bwd_t_group):
        t = t_block * bwd_t_group
        b = b_block * bwd_b_group
        s = ti_back_matf(((0.0,) * bwd_t_group,)*bwd_b_group)
        gwk_pad = ti.simt.block.SharedArray((fwd_b_group, Tmax), ti.f32)
        w_pad = ti.simt.block.SharedArray((Tmax,), ti.f32)
        for bi in ti.static(range(bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                    gwk_pad[bi, t + i] = gwk[b + bi, c, t + i]
                    if bi == 0:
                        w_pad[t + i] = w[c, t + i]
        ti.simt.block.sync()
        for bi in ti.static(range(0, bwd_b_group)):
            for u in range(0, t+1):
                for i in ti.static(range(0, bwd_t_group)):
                    s[bi, i] += gwk_pad[bi, (T-1)-(t+i-u)] * k[b + bi, c, u]
        ti.simt.block.sync()
        # The last triangle is specialized
        # u is replaced with t+1+j
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(1, bwd_t_group)):
                for j in ti.static(range(i)):
                    s[bi, i] += gwk_pad[bi, T-i+j] * k[b + bi, c, t+1+j]
        # write out
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                gw[b + bi, c, t+i] = s[bi, i]

        s = ti_back_matf(((0.0,) * bwd_t_group,)*bwd_b_group)
        # The first triangle is specialized
        # t' = t + i
        # u' = t' + j
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(0, bwd_t_group-1)):
                for j in ti.static(range(i, bwd_t_group-1)):
                    s[bi, i] += gwk_pad[bi, T+i-j-1] * w_pad[t+j]

        for bi in ti.static(range(0, bwd_b_group)):
            for u in range(t+bwd_t_group-1, T):
                for i in ti.static(range(0, bwd_t_group)):
                    s[bi, i] += gwk_pad[bi, (T-1)+(t+i-u)] * w_pad[u]
        ti.simt.block.sync()
        # write out
        for bi in ti.static(range(0, bwd_b_group)):
            for i in ti.static(range(bwd_t_group)):
                gk[b+bi, c, t+i] = s[bi, i]
