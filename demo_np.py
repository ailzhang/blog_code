import numpy as np
import math
import pdb
from utils import save_to_image
ivec2 = lambda x, y: np.array([x, y], dtype=int)

tile_width = 32
tile_height = 16
pw = 128
ph = 128
sx = 16
shift_x = ivec2(tile_width, 0)
shift_y = ivec2(sx, tile_height)

tile = np.arange(0, tile_width * tile_height).reshape(tile_height, tile_width)
tile = tile.astype(float) / (tile_height * tile_width)
image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph


def coord_to_tile_pixel(arr):
    arr[:, :, 1] = tile_height -1 - arr[:, :, 1]
    arr = np.flip(arr, 2)
    return arr

def image_pixel_to_coord(arr):
    arr[:, :, 0] = image_height - 1 + ph - arr[:, :, 0]
    arr[:, :, 1] -= pw
    arr = np.flip(arr, 2)
    return arr

def map_coord(arr):
    v = np.floor(arr[:, :, 1] / tile_height).astype(int)
    u = np.floor((arr[:, :, 0] - v * shift_y[0]) / tile_width).astype(int)
    uu = np.stack((u, u), axis=2)
    vv = np.stack((v, v), axis=2)
    return arr - uu * shift_x - vv * shift_y

def map_pixel(arr):
    arr1 = image_pixel_to_coord(arr)    
    arr2 = map_coord(arr1)
    return coord_to_tile_pixel(arr2)

image_pixels = np.zeros((image_height, image_width, 3), dtype=float)
nrows = np.arange(image_height)
ncols = np.arange(image_width)
cy, cx = np.meshgrid(ncols, nrows)
coords = np.stack((cx, cy), axis=2)
table = map_pixel(coords)
pdb.set_trace()
for row in range(image_height):
    for col in range(image_width):
        x, y = table[row, col]
        r = math.sin(math.pi * tile[x, y])
        g = math.cos(math.pi * tile[x, y] / 2)
        image_pixels[row, col] = [r, g, 0]

image_pixels = (255 * image_pixels).astype(np.uint8)
from PIL import Image
Image.fromarray(image_pixels).save("result.png")
