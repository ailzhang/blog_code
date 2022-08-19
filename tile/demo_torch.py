import numpy as np
import torch
from utils import save_to_image, generate_tile_image, Timer, tile_height, tile_width, pw, ph, sx, N

device = 'cuda'
shift_x = torch.tensor([tile_width, 0], device=device)
shift_y = torch.tensor([sx, tile_height], device=device)

tile = generate_tile_image(tile_height, tile_width)

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph

image_pixels = torch.zeros((image_height, image_width, 3),
                           device=device,
                           dtype=float)
nrows = torch.arange(image_height, device=device)
ncols = torch.arange(image_width, device=device)
cy, cx = torch.meshgrid(ncols, nrows)
coords = torch.stack((cx.T, cy.T), axis=2)
y = torch.tensor(tile.stride(), device=device, dtype=torch.float)


def torch_pad(arr, tile, y):
    # image_pixel_to_coord
    arr[:, :, 0] = image_height - 1 + ph - arr[:, :, 0]
    arr[:, :, 1] -= pw
    arr1 = torch.flip(arr, (2, ))
    
    # map_coord
    v = torch.floor(arr1[:, :, 1] / tile_height).to(torch.int)
    u = torch.floor((arr1[:, :, 0] - v * shift_y[0]) / tile_width).to(torch.int)
    uu = torch.stack((u, u), axis=2)
    vv = torch.stack((v, v), axis=2)
    arr2 = arr1 - uu * shift_x - vv * shift_y

    # coord_to_tile_pixel    
    arr2[:, :, 1] = tile_height - 1 - arr2[:, :, 1]
    table = torch.flip(arr2, (2, ))
    table = table.view(-1, 2).to(torch.float)
    inds = table.mv(y)
    gathered = torch.index_select(tile.view(-1), 0, inds.to(torch.long))

    return gathered

# Warmup: PyTorch code dramatically slows down when GPU RAM hits its limit 
# so it actually need a bit tweak to find the best runs.
for _ in range(3):
    gathered = torch_pad(coords, tile, y)
    gathered.zero_()

# with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
for _ in range(N):
    with Timer():
        gathered = torch_pad(coords, tile, y)
        torch.cuda.synchronize(device=device)
# print(prof.key_averages().table(sort_by="self_cuda_time_total"))

save_to_image(gathered.reshape(image_height, image_width), 'out_torch')
