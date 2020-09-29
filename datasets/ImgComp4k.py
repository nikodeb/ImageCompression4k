from .base import AbstractDataset
import imageio
import numpy as np


class ImgComp4KDataset(AbstractDataset):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.img_name = self.args.img_name

    @classmethod
    def code(cls):
        return 'samples'

    @classmethod
    def url(cls):
        return ''

    def get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_{}'.format(self.code(), self.img_name)
        return preprocessed_root.joinpath(folder_name)

    def preprocess(self):
        # retrieve raw image
        full_image_name = self.img_name if self.img_name.endswith('.jpg') else self.img_name + '.jpg'
        raw_image_path = self._get_rawdata_folder_path().joinpath(full_image_name)
        img = imageio.imread(raw_image_path)

        # convert to an array of rgb values
        img = np.asarray(img)
        rgb = np.reshape(img, (-1, 3))

        # create an array of 2D coordinates
        x_ind, y_ind = np.indices(img.shape[:-1])
        x_ind = np.reshape(x_ind, (-1, 1))
        y_ind = np.reshape(y_ind, (-1, 1))
        coords = np.hstack((x_ind, y_ind))

        # create dataset
        dataset = {'coords': coords,
                   'rgb': rgb}

        return dataset