import taichi as ti
import torch
import numpy as np
from utils import Timer, save_to_image, generate_tile_image, tile_height, tile_width, pw, ph, sx, N

ti.init(arch=ti.gpu, kernel_profiler=True)
torch_device = 'cuda'

shift_x = (tile_width, 0)
shift_y = (sx, tile_height)

tile = generate_tile_image(tile_height, tile_width, torch_device)

image_width = tile_width + 2 * pw
image_height = tile_height + 2 * ph

image_pixels = torch.zeros((image_height, image_width),
                           device=torch_device,
                           dtype=torch.float)



@ti.kernel
def ti_pad(image_pixels: ti.types.ndarray(), tile: ti.types.ndarray()):
    for row, col in ti.ndrange(image_height, image_width):
        # image_pixel_to_coord: maps a pixel in the large image to its coordinates (x, y) relative to the origin.
        # (which is the lower left corner of the tile image)
        x1, y1 = ti.math.ivec2(col - pw, image_height - 1 - row + ph)       
        # map_coord: let (x, y) = (x0, y0) + shift_x * u + shift_y * v.
        # This function finds P = (x0, y0) which is the corresponding point in the tile.
        v: ti.i32 = ti.floor(y1 / tile_height)
        u: ti.i32 = ti.floor((x1 - v * shift_y[0]) / tile_width)
        x2, y2 = ti.math.ivec2(x1 - u * shift_x[0] - v * shift_y[0],
                 y1 - u * shift_x[1] - v * shift_y[1])
        # coord_to_tile_pixel: The origin is located at the lower left corner of the tile image.
        # Assuming a point P with coordinates (x, y) lies in this tile, this function
        # maps P's coordinates to its actual pixel location in the tile image.
        x, y = ti.math.ivec2(tile_height - 1 - y2, x2)       
        image_pixels[row, col] = tile[x, y]


# Run once to compile pad kernel
ti_pad(image_pixels, tile)
image_pixels.fill_(0)

ti.profiler.clear_kernel_profiler_info()
for _ in range(N):
    with Timer():
        ti_pad(image_pixels, tile)
        ti.sync()
ti.profiler.print_kernel_profiler_info()

save_to_image(image_pixels, 'out_taichi')
