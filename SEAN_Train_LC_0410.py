import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from SEAN_Generator_LC_0409 import SEAN_Generator
from SEAN_Discriminator_LC_0409 import SEAN_Discriminator, VGG19
from SEAN_Dataset_LC_0410 import SEAN_Dataset

def Gan_Loss(pred_fake):

    loss = 0

    for pred_i in pred_fake:
        pred_i = pred_i[-1]
        loss_ =  -torch.mean(pred_i)
        bs = 1 if len(loss_.size()) == 0 else loss.size(0)
        new_loss = torch.mean(loss_.view(bs, -1), dim=1)
        loss += new_loss

    return loss / len(pred_fake)

def Gan_Feat_Loss(pred_fake, pred_real, featur_loss):

    gan_feat_loss = 0

    for i in range(2):  # 2: 判别器的个数
        for j in range(len(pred_fake) - 1):  # 判别器的最后一层是最终的预测结果，所以去掉 == 3
            unweighted_loss = featur_loss(pred_fake[i][j], pred_real[i][j].detach())
            gan_feat_loss += unweighted_loss * 10 / 2 # 最后求的是平局，给的权重是10

    return gan_feat_loss


class Vgg_Loss(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss * 10

def Dis_Loss(pred_fake, target_is_real):

    loss = 0
    for pred_i in pred_fake:
        pred_i = pred_i[-1]
        if target_is_real:
            minval = torch.min(pred_i - 1, get_zero_tensor(pred_fake))
            loss_ = -torch.mean(minval)
        else:
            minval = torch.min(-pred_i - 1, get_zero_tensor(pred_fake))
            loss_ = -torch.mean(minval)

        bs = 1 if len(loss_.size()) == 0 else loss.size(0)
        new_loss = torch.mean(loss_.view(bs, -1), dim=1)
        loss += new_loss

    return loss / len(pred_fake)

def get_zero_tensor(input):

    zero_tensor = torch.FloatTensor(1).fill_(0)
    zero_tensor.requires_grad_(False)

    return zero_tensor.expand_as(input)


if __name__ == "__main__":
    path = "/home/liuchao/SEAN-master/datasets/CelebA-HQ/train/labels"

    batch_size, numworkers, epoch = 0, 0, 0

    dataset = SEAN_Dataset(data_path=path)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=numworkers)

    Gen_Net = SEAN_Generator()
    Dis_Net = SEAN_Discriminator()

    optimizer_G = torch.optim.Adam(Gen_Net.paramters(), lr=0.0001, betas=(0, 0.9))
    optimizer_D = torch.optim.Adam(Dis_Net.paramters(), lr=0.0001, betas=(0, 0.9))

    feture_loss = torch.nn.L1Loss()
    vgg_loss = Vgg_Loss()

    for _ in range(epoch):

        for i, (segm, img) in enumerate(dataloader):

            fake_img = Gen_Net(img, segm)
            pred_fake, pred_real = Dis_Net(segm, img, fake_img)

            G_loss = Gan_Loss(pred_fake)
            G_Feat_Loss = Gan_Feat_Loss(pred_fake, pred_real, feture_loss)
            G_Vgg_Loss = vgg_loss(pred_fake, pred_real)

            G_Losses = {"GAN_loss":G_loss, "GAN_Feat_loss":G_Feat_Loss, "GAN_Vgg_loss": vgg_loss}
            mean_g_loss = sum(G_Losses.values()).mean()

            optimizer_G.zero_grad()
            mean_g_loss.backward()
            optimizer_G.step()

            D_Losses = {}
            with torch.no_grad():
                fake_image = Gen_Net(img, segm)
                fake_image = fake_image.detach()
                fake_image.requires_grad_()

            pred_fake, pred_real = Dis_Net(segm, img, fake_img)

            D_Losses['D_Fake'] = Dis_Loss(pred_fake, False)
            D_Losses['D_real'] = Dis_Loss(pred_real, True)

            mean_d_loss = sum(D_Losses.values()).mean()

            optimizer_D.zero_grad()
            mean_d_loss.backward()
            optimizer_D.step()




