import time
import numpy as np
import math
import pdb
from PIL import Image
import torch

tile_width = 32
tile_height = 48
pw = 4096
ph = 2048
sx = 10
N = 8

class Timer:
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *exc_info):
        print(f'Elapsed {time.time() - self.start} seconds')


def save_to_image(image_pixels, name):
    r = torch.sin(math.pi * image_pixels)
    g = torch.cos(math.pi * image_pixels / 2)
    b = torch.zeros_like(image_pixels)
    image_pixels = torch.stack([r, g, b], axis=2)
    image_pixels = (255 * image_pixels.cpu().numpy()).astype(np.uint8)
    Image.fromarray(image_pixels).save(f"imgs/{name}.png")


def generate_tile_image(tile_height, tile_width, device='cuda'):
    tile = torch.arange(0,
                        tile_width * tile_height,
                        dtype=torch.float,
                        device=device).reshape(tile_width, tile_height)
    tile = torch.rot90(tile, -1, (0, 1)).contiguous()
    tile = tile / (tile_height * tile_width)
    save_to_image(tile, 'orig')
    return tile
