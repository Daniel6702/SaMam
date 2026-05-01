

import torch.nn as nn
import torch
from LOSS.loss_util import normal, calc_mean_std

from pytorch_msssim import ssim

import os
loss_dir = os.path.dirname(__file__)
vgg_ckp = os.path.join(loss_dir, 'vgg_ckp', 'vgg_normalised.pth')

class Integration_loss(nn.Module):
    def __init__(self, apply_huber_loss = False, apply_SSIM_loss = False, apply_identity_loss=False):

        super().__init__()


        encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        encoder.load_state_dict(torch.load(vgg_ckp))
        encoder = nn.Sequential(*list(encoder.children())[:44])

        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        if apply_huber_loss:
            self.loss = nn.HuberLoss()  #nn.MSELoss()
        else:
            self.loss = nn.MSELoss()

        self.apply_SSIM_loss = apply_SSIM_loss
        self.apply_identity_loss = apply_identity_loss

    def calc_ssim_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
    
        # adjust if your range is [-1,1]
        input = torch.clamp(input, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
    
        return 1.0 - ssim(input, target, data_range=1.0, size_average=True)

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.loss(input_mean, target_mean) + self.loss(input_std, target_std)

    def forward(self, Ics, samples_cc, samples_ss, samples_c, samples_s):
        Ic_feats = self.encode_with_intermediate(samples_c)
        Is_feats = self.encode_with_intermediate(samples_s)
    
        Ics_feats = self.encode_with_intermediate(Ics)
    
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(Ic_feats[-1])) + \
                 self.calc_content_loss(normal(Ics_feats[-2]), normal(Ic_feats[-2]))
    
        loss_s = self.calc_style_loss(Ics_feats[0], Is_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], Is_feats[i])
    
        if self.apply_identity_loss:
            loss_lambda1 = self.calc_content_loss(samples_cc, samples_c) + \
                           self.calc_content_loss(samples_ss, samples_s)
    
            Icc_feats = self.encode_with_intermediate(samples_cc)
            Iss_feats = self.encode_with_intermediate(samples_ss)
    
            loss_lambda2 = self.calc_content_loss(Icc_feats[0], Ic_feats[0]) + \
                           self.calc_content_loss(Iss_feats[0], Is_feats[0])
    
            for i in range(1, 5):
                loss_lambda2 += self.calc_content_loss(Icc_feats[i], Ic_feats[i]) + \
                                self.calc_content_loss(Iss_feats[i], Is_feats[i])
        else:
            loss_lambda1 = Ics.new_tensor(0.0)
            loss_lambda2 = Ics.new_tensor(0.0)
    
        if self.apply_SSIM_loss:  
            '''
            SSIM loss is now applied to the identity paths instead of the (Ics, samples_c)
            '''
            loss_ssim = (
                self.calc_ssim_loss(samples_cc, samples_c) +
                self.calc_ssim_loss(samples_ss, samples_s)
            )
        else:
            loss_ssim = Ics.new_tensor(0.0)
    
        return loss_c, loss_s, loss_lambda1, loss_lambda2, loss_ssim

if __name__ == '__main__':

    net = Integration_loss().cuda()

    print('# net parameters:', sum(param.numel() for param in net.parameters()), '\n')

    Ics = torch.randn((4, 3, 256, 256)).cuda()
    Ic = torch.randn((4,3,256,256)).cuda()
    Icc = torch.randn((4,3,256,256)).cuda()
    Is = torch.randn((4,3,256,256)).cuda()
    Iss = torch.randn((4,3,256,256)).cuda()
    out = net.forward(Ics, Icc,Iss,Ic,Is)
    print(out)


