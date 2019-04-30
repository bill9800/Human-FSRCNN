from __future__ import print_function
from math import log10
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append('./')
from model import Net
import os
import numpy as np

class FSRCNNTrainer(object):
    def __init__(self, config, training_loader, testing_loader):
        super(FSRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.CUDA else 'cpu')
        self.model = None
        self.lr = config.lr
        self.nEpochs = config.nEpochs
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed
        self.upscale_factor = config.upscale_factor
        self.training_loader = training_loader
        self.testing_loader = testing_loader
        self.testpsnr = -1

    def build_model(self):
        self.model = Net(num_channels=1, upscale_factor=self.upscale_factor).to(self.device)
        self.model.weight_init(mean=0.0, std=0.2)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[5, 7, 9], gamma=0.5)  # lr decay


    def save_model(self):
        model_out_path = "model_path.pth"
        torch.save(self.model, model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for batch_num, (data, target) in enumerate(self.training_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(data), target)
            train_loss += loss.item()
            # print(train_loss / (batch_num + 1))
            loss.backward()
            self.optimizer.step()
            print('batch_num:', batch_num, '/', len(self.training_loader), 'Loss:', train_loss / (batch_num + 1))
            # progress_bar(batch_num, len(self.training_loader), 'Loss: %.4f' % (train_loss / (batch_num + 1)))

        print("    Average Loss: {:.4f}".format(train_loss / len(self.training_loader)))

    def test(self):
        self.model.eval()
        avg_psnr = 0
        avg_psnr2 = 0
        avg_psnr3 = 0
        avg_ssim = 0

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                # psnr
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
                # ssim
                avg_ssim += ssim(target,prediction)
                # psnr test2
                avg_psnr2 += psnr(target,prediction)
                # psnr test3
                mse2 = np.mean((target-prediction)**2)
                PIXEL_MAX = max(target.shape[0],target.shape[1])
                avg_psnr3 += 10*log10(PIXEL_MAX/mse2)
            print('test PSNR:', avg_psnr / len(self.testing_loader))
            print('test PSNR2:', avg_psnr2 / len(self.testing_loader))
            print('test PSNR3:', avg_psnr3	/ len(self.testing_loader))
            print('test SSIM:', avg_ssim / len(self.testing_loader))
            # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        save_path = './saved_fsrcnn_face_crop_32_0.001'
        if avg_psnr / len(self.testing_loader) > self.testpsnr:
            self.testpsnr = avg_psnr / len(self.testing_loader)
            folder = os.path.exists(save_path)
            if not folder:
                os.makedirs(save_path)
                print('create folder to save models')
            torch.save(self.model, save_path + '/model_' + str(self.testpsnr) + '.pkl')
            print('model saved ......')
        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        print('Total parameters:', sum(p.numel() for p in self.model.parameters()))
        print('Total trainable parameters:', sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
