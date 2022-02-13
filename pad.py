import taichi as ti
import torch
from torchvision.utils import save_image

ti.init()
DIM = 3
@ti.func
def wrap_indices_single(indices: ti.template(), cell_vectors: ti.template()):
    indices_new = ti.Vector([indices[0], indices[1], indices[2]])
    for d in ti.static(range(DIM - 1, -1, -1)):
        offset = indices_new[d] // int(cell_vectors[d, d])
        indices_new = indices_new - int(offset * ti.Vector([
            cell_vectors[0, d],
            cell_vectors[1, d],
            cell_vectors[2, d],
        ]))
    #if indices_new.x < 0 or indices_new.y < 0 or indices_new.z < 0 \
    #    or indices_new.x >= cell_vectors[0, 0] or indices_new.y >= cell_vectors[1, 1] or indices_new.z >= cell_vectors[2, 2]:
    #    print(indices, indices_new, offset_all, cell_vectors[0, 0], cell_vectors[1, 1], cell_vectors[2, 2])
    return indices_new

@ti.kernel
def pad_kernel(y: ti.any_arr(), cell_vectors: ti.any_arr(), pw: int, ph: int, pd: int):
    """
    Shapes:
        x: (b * c * w * h * d)
        y: (b * c * (w + 2 * pw) * (h + 2 * ph) * (d + 2 * pd))
        cell_vectors: (b * 3 * 3)
    """
    w, h, d = y.shape[2] - 2 * pw, y.shape[3] - 2 * ph, y.shape[4] - 2 * pd
    n_pad_d = (w + 2 * pw) * (h + 2 * ph) * 2 * pd
    n_pad_h = (w + 2 * pw) * d * 2 * ph
    n_pad_w = h * d * 2 * pw
    pad_vector = ti.Vector([pw, ph, pd])
    for b, idx in ti.ndrange(y.shape[0], n_pad_d + n_pad_h + n_pad_w):
        i, j, k = 0, 0, 0
        if idx < n_pad_d:
            # depth padding
            i = idx // (2 * pd) // (h + 2 * ph)
            j = idx // (2 * pd) % (h + 2 * ph)
            k = idx % (2 * pd)
            k = k + d if k >= pd else k
        elif idx < n_pad_d + n_pad_h:
            # height padding:
            i = (idx - n_pad_d) // (2 * ph) // d
            j = (idx - n_pad_d) % (2 * ph)
            j = j + h if j >= ph else j
            k = (idx - n_pad_d) // (2 * ph) % d + pd
        else:
            # width padding
            i = (idx - n_pad_d - n_pad_h) % (2 * pw)
            i = i + w if i >= pw else i
            j = (idx - n_pad_d - n_pad_h) // (2 * pw) // d + ph
            k = (idx - n_pad_d - n_pad_h) // (2 * pw) % d + pd
        indices = ti.Vector([i, j, k])
        indices_new = wrap_indices_single(indices - pad_vector, cell_vectors) + pad_vector
        # pad by each channel
        for c in range(y.shape[1]):
            y[b, c, i, j, k] = y[b, c, indices_new.x, indices_new.y, indices_new.z]

# input_file = 'hexagon.png'
# input_image = ti.imread(input_file)
# import pdb
# pdb.set_trace()
# ti.tools.image.imwrite(input_image, 'copy.png')
b = 1
c = 3
# a = torch.range(0, 8).reshape(3, 3)
a = torch.ones(3, 3)
y = torch.range(0, 647).reshape(b, c, 6, 6, 6)
pad_kernel(y, a, 0, 0, 0)
# save_image(y, 'copied.png')
print(y)

