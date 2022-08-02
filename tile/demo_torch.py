import numpy as np
import math
import pdb
import torch
from utils import save_to_image, generate_tile_image, Timer

device = 'cuda'
tile_width = 32
tile_height = 48
pw = 128
ph = 128
sx = 16
shift_x = torch.tensor([tile_width, 0], device=device)
shift_y = torch.tensor([sx, tile_height], device=device)

tile = generate_tile_image(tile_height, tile_width, device)
save_to_image(tile, 'orig_torch')

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph


def coord_to_tile_pixel(arr):
    arr[:, :, 1] = tile_height - 1 - arr[:, :, 1]
    arr = torch.flip(arr, (2, ))
    return arr


def image_pixel_to_coord(arr):
    arr[:, :, 0] = image_height - 1 + ph - arr[:, :, 0]
    arr[:, :, 1] -= pw
    arr = torch.flip(arr, (2, ))
    return arr


def map_coord(arr):
    v = torch.floor(arr[:, :, 1] / tile_height).to(torch.int)
    u = torch.floor((arr[:, :, 0] - v * shift_y[0]) / tile_width).to(torch.int)
    uu = torch.stack((u, u), axis=2)
    vv = torch.stack((v, v), axis=2)
    return arr - uu * shift_x - vv * shift_y


def map_pixel(arr):
    arr1 = image_pixel_to_coord(arr)
    arr2 = map_coord(arr1)
    return coord_to_tile_pixel(arr2)


image_pixels = torch.zeros((image_height, image_width, 3),
                           device=device,
                           dtype=float)
nrows = torch.arange(image_height, device=device)
ncols = torch.arange(image_width, device=device)
cy, cx = torch.meshgrid(ncols, nrows)
coords = torch.stack((cx.T, cy.T), axis=2)
y = torch.tensor(tile.stride(), device=device, dtype=torch.float)

with Timer():
    table = map_pixel(coords)
    table = table.view(-1, 2).to(torch.float)
    inds = table.mv(y)
    gathered = torch.index_select(tile.view(-1), 0, inds.to(torch.long))

save_to_image(gathered.reshape(image_height, image_width), 'torch')
