import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SEAN_Discriminator(nn.Module):

    def __init__(self, D_num=2):
        super().__init__()
        self.D_num = D_num # 判别器的个数
        for i in range(1, D_num+1):
            D_net = Multiscale_Disciminator()
            self.add_module("discriminator_{}".format(i), D_net)

        """将图片大小减半，再传入下一级判别器"""
        # self.down_sampling = F.avg_pool2d(input, kernel_size=3,stride=2, padding=1,count_include_pad=False)
        # self.down_sampling = nn.AvgPool2d(kernel_size=3,stride=2, padding=1,count_include_pad=False)

    def forward(self, orig_segm, real_img, fake_img):
        """
        :param orig_segm: 分割图 N*19*256*256
        :param real_img: 原图片 N*3*256*256
        :param fake_img: 生成器生成的图片 N*3*256*256
        :return: result: list [disciminator1_out, disciminator2_out, ...]
        """

        fake_concat = torch.cat([orig_segm, fake_img], dim=1)
        real_concat = torch.cat([orig_segm, real_img], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0) # 将生成的图片和真实的图片放在一个批次训练

        result = []
        fake = []
        real = []
        """
        如：                               net
                       net_0          net_1      net_2      ...     net_n
                net_0_0 ... net_0_n       ....                  net_n_0 ... net_n_n
                   .                                                    .
                   .                                                    .
                   .                                                    .
        self.children(): 返回该模型下一层的子模型    # if net: return net_0, ..., net_n; if net_0: return net_0_0, ..., net_0_n
        self.modules(): 返回该模型的所有子模型以及本身 # if net: retrun all; if net_0: retrun net_0_all
        self.named_children(): 返回子模块的迭代器 # 与self.children()相同，只是调用方式不一样，如下所示
        self.named_modules(): 返回所有模块的迭代器 
        """
        """方式一"""
        for name, net_d in self.named_children():
            print(name)
            out = net_d(fake_and_real)
            result.append(out)
            fake_and_real = self.down_sampling(fake_and_real)

        """方式二"""
        # for _, (name, net_d) in enumerate(self.named_children()):
        #
        #     out = net_d(fake_and_real)
        #     result.append(out)
        #
        #     fake_and_real = self.down_sampling(fake_and_real)

        for p in result:
            fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
            real.append([tensor[tensor.size(0) // 2:] for tensor in p])

        return result, fake, real

    def down_sampling(self, input):

        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=1, count_include_pad=False)


class Multiscale_Disciminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.Conv_1 = nn.Sequential(

            nn.Conv2d(in_channels=22, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.Conv_2 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=2, bias=False)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.Conv_3 = nn.Sequential(

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2, bias=False)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.Conv_4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=4, stride=1, padding=2)

    def forward(self, input):

        out = []

        self.out_1 = self.Conv_1(input)
        self.out_2 = self.Conv_2(self.out_1)
        self.out_3 = self.Conv_3(self.out_2)
        self.out_4 = self.Conv_4(self.out_3)

        out.extend([self.out_1, self.out_2, self.out_3, self.out_4])

        return out

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out



x = SEAN_Discriminator(2)

a = torch.randn((1,19, 256, 256))
b = torch.randn((1,3, 256, 256))
c = torch.randn((1,3, 256, 256))

result, fake, real = x(a, b, c)
# print(len(result))
print(len(fake))
print(len(fake[0]))

print(fake[0][0].size())

# print(len(result[1]))
# print(result[1][0].size())

