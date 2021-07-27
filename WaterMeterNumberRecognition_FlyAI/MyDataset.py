import os

from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import random

DATA_PATH = 'data/input/WaterMeterNumberRecognition/'
CROP_PATH = 'data/input/images/'


class MyDataset(Dataset):
    def __init__(self, input_x, input_y, transform=None):
        self.file_names = input_x
        self.label = input_y
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        path = os.path.join(CROP_PATH, self.file_names[idx])
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.label[idx]

        return image, int(label)


if __name__ == '__main__':
    import matplotlib.pylab as plt
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    path = os.path.join(DATA_PATH, './image/4494.jpg')
    print(DATA_PATH, path)
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt

    img = cv2.imread(path)
    dst = cv2.fastNlMeansDenoising(img, None, 16, 10, 7)
    plt.subplot(121), plt.imshow(img)
    plt.subplot(122), plt.imshow(dst)
    plt.show()
    # image = Image.open(path).convert("RGB")
    #
    #
    # def crop_picture_for_prediction(image):
    #     width, height = image.size
    #     cropped_images = []
    #     for left in range(5):
    #         box = (round(left * width / 5), 0, round((left + 1) * width / 5), height)
    #         region = image.crop(box)
    #         cropped_images.append(region)
    #     return cropped_images
    #
    #
    # images = crop_picture_for_prediction(image)
    # for img in images:
    #     print(img.size)
    #     plt.figure()
    #     plt.imshow(img)
    # plt.show()