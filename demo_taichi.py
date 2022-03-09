import taichi as ti
import torch
import math
import numpy as np
from utils import Timer, save_to_image, generate_tile_image

ti.init(arch=ti.cpu)
device = 'cuda'

tile_width = 32
tile_height = 32
pw = 128
ph = 128
sx = 10

ivec2 = lambda x, y: ti.Vector([x, y], dt=ti.i32)

shift_x = ivec2(tile_width, 0)
shift_y = ivec2(sx, tile_height)

tile = generate_tile_image(tile_height, tile_width, device)

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph

image_pixels = torch.zeros((image_height, image_width, 3), device=device, dtype=torch.float)


@ti.func
def image_pixel_to_coord(x, y):
    """Map a pixel in the large image to its coordinates (x, y) relative to the origin.
    (which is the lower left corner of the tile image)
    """
    return ivec2(y - pw, image_height - 1 - x + ph)


@ti.func
def map_coord(x, y):
    """Assume (x, y) = (x0, y0) + shift_x * u + shift_y * v.
    This function finds P = (x0, y0) which is the corresponding point in the tile.
    """
    v = ti.floor(y / tile_height)
    u = ti.floor((x - v * shift_y[0]) / tile_width)
    return ivec2(x, y) - u * shift_x - v * shift_y


@ti.func
def map_pixel(x, y):
    x1, y1 = image_pixel_to_coord(x, y)
    x2, y2 = map_coord(x1, y1)
    return coord_to_tile_pixel(x2, y2)


@ti.func
def coord_to_tile_pixel(x, y):
    """The origin is located at the lower left corner of the tile image.
    Assuming a point P with coordinates (x, y) lies in this tile, this function
    maps P's coordinates to its actual pixel location in the tile image.
    """
    return ivec2(tile_height - 1 - y, x)


@ti.kernel
def pad(image_pixels: ti.ext_arr(), tile: ti.any_arr()):
    for row, col in ti.ndrange(image_height, image_width):
        x, y = map_pixel(row, col)
        x_int = ti.cast(x, ti.i32)
        y_int = ti.cast(y, ti.i32)
        image_pixels[row, col, 0] = tile[x_int, y_int, 0]
        image_pixels[row, col, 1] = tile[x_int, y_int, 1]
        image_pixels[row, col, 2] = tile[x_int, y_int, 2]

# Excludes Taichi JIT compilation
pad(image_pixels, tile)
# Reinitialize image_pixels to zeros
image_pixels = torch.zeros((image_height, image_width, 3), device=device, dtype=torch.float)

with Timer():
    pad(image_pixels, tile)

save_to_image(image_pixels, 'taichi')
