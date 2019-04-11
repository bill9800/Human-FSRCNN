from os import listdir
from os.path import join

import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_image = load_img(self.image_filenames[index])
        target = input_image.copy()
        # print(target.size)
        if self.input_transform:
            input_image = self.input_transform(input_image)
        else:
            transi = Compose([
                            Resize((input_image.size[1] // 4, input_image.size[0] // 4), Image.BICUBIC),
                            ToTensor()
                            ])
            input_image = transi(input_image)
        if self.target_transform:
            target = self.target_transform(target)
        else:
            transt = Compose([
                ToTensor()
            ])
            target = transt(target)

        # print(input_image.size(), target.size())
        return input_image, target

    def __len__(self):
        return len(self.image_filenames)
