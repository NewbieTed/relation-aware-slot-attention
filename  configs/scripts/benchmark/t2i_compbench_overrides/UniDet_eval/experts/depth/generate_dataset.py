# Copyright (c) 2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/prismer/blob/main/LICENSE

import glob

from torch.utils.data import Dataset
from PIL import ImageFile
from dataset.utils import *
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Dataset(Dataset):
    def __init__(self, data_path, transform, sample_paths=None):
        self.data_path = data_path
        self.transform = transform

        if sample_paths is not None:
            self.data_list = list(sample_paths)
            self.data_list.sort(
                key=lambda path: int(os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[1])
            )
            return

        data_images = os.path.join(data_path, "samples")
        data_imgs = [
            name for name in os.listdir(data_images)
            if not name.startswith(".") and os.path.isfile(os.path.join(data_images, name))
        ]
        data_imgs.sort(key=lambda x: int(os.path.splitext(x)[0].rsplit("_", 1)[1]))
        self.data_list = [os.path.join(data_images, data) for data in data_imgs]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path = self.data_list[index]
        image = Image.open(image_path).convert('RGB')
        img_size = [image.size[0], image.size[1]]
        image = self.transform(image)
        return image, image_path, img_size
