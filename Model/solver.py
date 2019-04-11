from __future__ import print_function
from math import log10

import torch
import torch.backends.cudnn as cudnn
import sys
sys.path.append('./')
from model import Net
import os


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
        self.testpsnr = 0

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
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)  # lr decay

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

        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.testing_loader):
                data, target = data.to(self.device), target.to(self.device)
                prediction = self.model(data)
                mse = self.criterion(prediction, target)
                psnr = 10 * log10(1 / mse.item())
                avg_psnr += psnr
        print('test PSNR:', avg_psnr / len(self.testing_loader))
                # progress_bar(batch_num, len(self.testing_loader), 'PSNR: %.4f' % (avg_psnr / (batch_num + 1)))

        save_path = './saved_models'
        if avg_psnr / len(self.testing_loader) > self.testpsnr:
            self.testpsnr = avg_psnr / len(self.testing_loader)
            folder = os.path.exists(save_path)
            if not folder:
                os.makedirs(save_path)
                print('create folder to save models')
            torch.save(self.model, save_path + '/model_' + str(self.testpsnr))
            print('model saved ......')
        print("    Average PSNR: {:.4f} dB".format(avg_psnr / len(self.testing_loader)))

    def run(self):
        self.build_model()
        for epoch in range(1, self.nEpochs + 1):
            print("\n===> Epoch {} starts:".format(epoch))
            self.train()
            self.test()
            self.scheduler.step(epoch)
