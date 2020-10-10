import imageio
import numpy as np
from pathlib import Path

import torch

raw_image_path = Path('Data/samples/1001.jpg')
img = imageio.imread(raw_image_path)

img = np.asarray(img)
img = np.moveaxis(img, -1, 0)
# rgb = np.reshape(img, (-1, 3))
#
# x_ind, y_ind = np.indices(img.shape[:-1])
# x = np.reshape(x_ind, (-1, 1))
# y = np.reshape(y_ind, (-1, 1))
# coords = np.hstack((x, y))
#
# print('coordinate 10,1000, rgb values '+str(img[10][1000]))
# print('coordinate 10,1000, rgb values '+str(rgb[(10*3840)+1000]))
#
# print('coordinate 10,1000: ' + str(x_ind[10][1000]) + ',' + str(y_ind[10][1000]))
# print('coordinate 10,1000: ' + str(coords[(10*3840)+1000]))
#

height = torch.linspace(-1, 1, steps=3)
width = torch.linspace(-1, 1, steps=2)
tensors = (height, width)
coords = torch.stack(torch.meshgrid(*tensors), dim=-1)
coords2 = coords.reshape(-1, 2)


print('x')