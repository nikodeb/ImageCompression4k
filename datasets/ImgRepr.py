from .base import AbstractDataset
import imageio
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, ToPILImage
import numpy as np
import torch


class ImgRepr4KDataset(AbstractDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.img_name = self.args.img_name
        self.img_resize_height = self.args.img_resize_height
        self.img_resize_width = self.args.img_resize_width
        self.normalise_coords = self.args.normalise_coords

    @classmethod
    def code(cls):
        return 'samples'

    @classmethod
    def url(cls):
        return ''

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_{}_{}_{}_{}' \
            .format(self.code(), self.img_name, self.img_resize_height, self.img_resize_width, self.normalise_coords)
        return preprocessed_root.joinpath(folder_name)

    def preprocess(self):
        # retrieve raw image
        full_image_name = self.img_name if self.img_name.endswith('.jpg') else self.img_name + '.jpg'
        raw_image_path = self._get_rawdata_folder_path().joinpath(full_image_name)
        img = imageio.imread(raw_image_path)
        img = np.asarray(img)

        # preprocess image and convert to an array of normalised [0..1] rgb values
        img, transf_info = self.transform_image(image=img)
        rgb = np.reshape(img, (-1, 3))

        # create an array of 2D coordinates
        coords = self.generate_coords(image=img)

        # create dataset
        train = {'coords': coords,
                 'rgb': rgb}
        dataset = {'train': train,
                   'val': None,
                   'test': None,
                   'height': self.img_resize_height,
                   'width': self.img_resize_width,
                   'transform_info': transf_info}
        return dataset

    def transform_image(self, image):
        transform = Compose([
            ToPILImage(),
            Resize((self.img_resize_height, self.img_resize_width)),  # resize image
            ToTensor()  # convert from [0..255](HxWxC) -> [0..1](CxHxW)
        ])
        img = transform(image)
        channel_means = img.view(3, -1).mean(dim=1)
        channel_stds = img.view(3, -1).std(dim=1)
        # normalise each rgb channel to zero mean and unit var
        img = Normalize(mean=channel_means, std=channel_stds, inplace=False)(img)

        other_info = {'means': channel_means,
                      'stds': channel_stds}
        return img, other_info

    def generate_coords(self, image):
        if self.normalise_coords == 'none':
            x_ind, y_ind = np.indices(image.shape[:-1])
            x_ind = np.reshape(x_ind, (-1, 1))
            y_ind = np.reshape(y_ind, (-1, 1))
            coords = np.hstack((x_ind, y_ind))
        elif self.normalise_coords == 'negpos1':
            height = torch.linspace(-1, 1, steps=self.img_resize_height)
            width = torch.linspace(-1, 1, steps=self.img_resize_width)
            tensors = ([height, width])
            coords = torch.stack(torch.meshgrid(*tensors), dim=-1)
            coords = coords.reshape(-1, 2)
            coords = coords.numpy()
        else:
            raise NotImplementedError('Value of normalise_coords is not supported: {}'.format(self.normalise_coords))
        return coords
