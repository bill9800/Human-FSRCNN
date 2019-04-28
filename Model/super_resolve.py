from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor
from math import log10
from model import Net
import skimage
import tensorflow as tf
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
from time import time

import numpy as np

# ===========================================================
# Argument settings
# ===========================================================
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input', type=str, required=False, default='../dataset/HR_img_test/0.jpg', help='input image to use')
parser.add_argument('--model', type=str, default='model_path2.pth', help='model file to use')
parser.add_argument('--output', type=str, default='test.jpg', help='where to save the output image')
args = parser.parse_args()
# print(args)


# ===========================================================
# input image setting
# ===========================================================
GPU_IN_USE = torch.cuda.is_available()
img = Image.open(args.input).convert('YCbCr')
y, cb, cr = img.split()


# ===========================================================
# model import & setting
# ===========================================================
device = torch.device('cuda' if GPU_IN_USE else 'cpu')
model = torch.load('./model_path2.pth')
model = model.to(device)
# data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
# data = data.to(device)

if GPU_IN_USE:
    cudnn.benchmark = True


# ===========================================================
# output and save image
# ===========================================================
for i in range(100):
    rgb_img = Image.open('../dataset/LR_img_0.25/' + str(i) + '.jpg')
    img = rgb_img.convert('YCbCr')

    fsrcnn_start = time()

    y, cb, cr = img.split()
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = data.to(device)
    out = model(data)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    # print(out_img_y.size)
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_fsrcnn = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    fsrcnn_end = time()

    out_fsrcnn.save('../test_out/' + str(i) + 'fsrcnn.jpg')

    bic_start = time()

    out_bic = rgb_img.resize((rgb_img.size[0] * 4, rgb_img.size[1] * 4), Image.BICUBIC)

    bic_end = time()
    out_bic.save('../test_out/' + str(i) + 'bic.jpg')

    target = ToTensor()(Image.open('../dataset/HR_img_4/' + str(i) + '.jpg').convert('YCbCr').split()[0])

    # y, _, _ = out_fsrcnn.convert('YCbCr').split()
    # transi = Compose([
    #     ToTensor()
    # ])
    # y = transi(y)
    # y = ToTensor()(y)
    # print(y)
    with torch.no_grad():
        fsrcnnmse = torch.nn.MSELoss()(ToTensor()(out_fsrcnn.convert('YCbCr').split()[0]), target)
        fsrcnnpsnr = 10 * log10(1 / fsrcnnmse.item())
        bicmse = torch.nn.MSELoss()(ToTensor()(out_bic.convert('YCbCr').split()[0]), target)
        bicpsnr = 10 * log10(1 / bicmse.item())

    # fsrcnnpsnr = skimage.measure.compare_psnr(target, out_fsrcnn)
    # bicpsnr = skimage.measure.compare_psnr(target, out_bic)

    # fsrcnnpsnr = tf.image.psnr(tf.image.decode_jpeg(target), tf.image.decode_jpeg(out_fsrcnn), 255)
    # bicpsnr = tf.image.psnr(target, out_bic, 255)

    print(str(i))
    print('fsrcnn:', fsrcnnpsnr, fsrcnn_end - fsrcnn_start)
    print('bic:', bicpsnr, bic_end - bic_start)
    print('')




# out = model(data)
# out = out.cpu()
# out_img_y = out.data[0].numpy()
# print(out_img_y.size)
# out_img_y *= 255.0
# out_img_y = out_img_y.clip(0, 255)
# out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
#
# out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
# out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
# out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
#
# out_img.save(args.output)
# print('output image saved to ', args.output)
