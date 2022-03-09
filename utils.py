import time
import numpy as np
import math
from PIL import Image
import torch


class Timer:
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, *exc_info):
        print(f'Elapsed {time.time() - self.start} seconds')


def save_to_image(image_pixels, name):
    image_pixels = (255 * image_pixels.cpu().numpy()).astype(np.uint8)
    Image.fromarray(image_pixels).save(f"{name}.png")


def generate_tile_image(tile_height, tile_width, device):
    tile = torch.arange(0, tile_width * tile_height * 3,
                        dtype=torch.float, device=device).reshape(tile_height, tile_width, 3)
    tile = tile / (tile_height * tile_width * 3)
    for row in range(tile_height):
        for col in range(tile_width):
            tile[row, col, 0] = math.sin(math.pi * tile[row, col, 0])
            tile[row, col, 1] = math.cos(math.pi * tile[row, col, 1] / 2)
            tile[row, col, 2] = 0
    return tile
