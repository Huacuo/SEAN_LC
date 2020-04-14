import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from batchnorm import SynchronizedBatchNorm2d

class StyleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampling = nn.Sequential(

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=0,stride=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=False),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=False)
        )

        self.upsampling = nn.Sequential(

            nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(256),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0, stride=1),
            nn.Tanh()
        )


    def forward(self, x, seg):

        self.down = self.downsampling(x) # 下采样
        # print(self.down.size()) # 1*128*64*64
        self.up = self.upsampling(self.down) # 上采样
        # print(self.up.size()) # 1*512*128*128

        seg = F.interpolate(seg, size=(128, 128), mode="nearest") # N * 19 * 128 * 128

        batch, s = x.size(0), seg.size(1)
        style_code = torch.zeros((batch, s, 512)) # N * 19 * 512

        for i in range(batch):
            for j in range(s):
                component_mask_area = torch.sum(seg.byte()[i, j])
                if component_mask_area > 0:
                    """
                    masked_select: 将输入张量按照mask取值，即：在input上取mask上为1所在的索引的值
                    这里取style_code的逻辑是：先把输入的分割图（one_hot形式，19 * 256 * 256）下采样成 19 * 128 * 128的seg，
                    以及输入图片（3 * 256 * 256）经过encoder_decoder模型提取特征后得到 512 * 128 *128 的code。
                    这里的code就可以理解成整张图片的所有风格style。然后根据seg上每一个style所在的区域，在code上面去把那一部分值取出来，
                    之后对这些code做一个平均池化（这里的平均池化跟往常的不一样， 因为style的区域是不规则的，所以就先把取出来的code reshape成
                    512 * component_mask_area大小，然后再在 1 维上取平均得到 512 的向量）。
                    component_mask_area: 表示该种style的总共有多少个像素。0：表示在分割图上没有这一类，因此就不用提取该类的风格；>0: 
                    该类存在分割图上，所以需要提取style_code
                    """
                    codes_component_feature = self.up[i].masked_select(seg.byte()[i, j]).reshape(512, component_mask_area).mean(1)
                    style_code[i, j] = codes_component_feature

        return style_code

class SEAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.Conv = nn.Conv2d(in_channels=19, out_channels=1024, kernel_size=3, padding=1, stride=1)
        self.style_code = StyleEncoder()

        self.ResBlk_0 = SEAN_ResBlk(1024, 1024)
        self.ResBlk_1 = SEAN_ResBlk(1024, 1024)
        self.ResBlk_2 = SEAN_ResBlk(1024, 1024)
        self.ResBlk_3 = SEAN_ResBlk(1024, 512)
        self.ResBlk_4 = SEAN_ResBlk(512, 256)
        self.ResBlk_5 = SEAN_ResBlk(256, 128)
        self.ResBlk_6 = SEAN_ResBlk(128, 64, use_ST=False)

        self.upsampling = nn.Upsample(scale_factor=2)

        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1)


    def forward(self, input, seg_orig):
        """

        :param input: 图片，N*3*256*256
        :param seg_orig: 分割图，N*19*256*256
        :return:
        """

        seg_conv = F.interpolate(seg_orig, size=(8, 8))  # 直接对分割图下采样成8*8大小 N*19*256*256 ===>> N*19*8*8
        seg_conv = self.Conv(seg_conv) # N * 1024 * 8 * 8
        style_code = self.style_code(input=input, segmap=seg_orig) # N * 19 * 512

        seg_conv = self.ResBlk_0(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_1(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_2(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_3(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_4(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_5(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.ResBlk_6(seg_orig, seg_conv, style_code)
        seg_conv = self.upsampling(seg_conv)

        seg_conv = self.out(F.leaky_relu(seg_conv, 2e-1))
        generate_out = F.tanh(seg_conv)

        return generate_out

class SEAN_ResBlk(nn.Module):

    def __init__(self, in_channel, out_channel, use_ST=True, channel_change=True, short_cut=True):
        super().__init__()
        self.use_ST = use_ST
        self.channel_change = channel_change
        self.short_cut = short_cut
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channel = min(self.in_channel, self.out_channel)

        self.res_blk_0 = Detail_ResBlk(self.in_channel)
        self.res_blk_1 = Detail_ResBlk(self.mid_channel)
        self.res_blk_s = Detail_ResBlk(self.in_channel)

        self.Conv_0 = nn.Conv2d(in_channels=self.in_channel, out_channels=self.mid_channel, kernel_size=3, padding=1, stride=1)
        self.Conv_1 = nn.Conv2d(in_channels=self.mid_channel, out_channels=self.out_channel, kernel_size=3, padding=1, stride=1)
        self.Conv_s = nn.Conv2d(self.in_channel, self.out_channel, kernel_size=1, bias=False)

        self.Conv_0 = spectral_norm(self.Conv_0)
        self.Conv_1 = spectral_norm(self.Conv_1)
        self.Conv_s = spectral_norm(self.Conv_s)


    def forward(self, seg_orig, seg_conv, style_code):

        self.res_blk_0_out = self.Conv_0(self.act(self.res_blk_0(seg_conv, seg_orig, style_code)))
        self.res_blk_1_out = self.Conv_1(self.act(self.res_blk_1(self.res_blk_0_out, seg_orig, style_code)))
        self.res_blk_s_out = self.Conv_s(self.res_blk_s(seg_conv, seg_orig, style_code))

        self.SEAN_ResBlk_out = self.res_blk_1_out + self.res_blk_s_out

        return self.SEAN_ResBlk_out

    def act(self, x):
        return F.leaky_relu(x, 2e-1)

class Detail_ResBlk(nn.Module):

    def __init__(self, channel_num, use_ST=True, channel_change=True, short_cut=True):
        super().__init__()

        self.use_ST = use_ST

        self.noise_var = nn.Parameter(torch.zeros(channel_num), requires_grad=True) # 学习噪声的参数
        self.blending_gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.blending_beta = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.norm = SynchronizedBatchNorm2d(channel_num, affine=False)
        self.per_style_convs = nn.ModuleList([nn.Linear(512, 512) for i in range(19)])

        self.conv_gamma = nn.Conv2d(512, channel_num, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(512, channel_num, kernel_size=3, padding=1)

        self.spade = SPADE(channel_num)


    def forward(self, seg_orig, seg_conv, style_code):

        noise = torch.randn(seg_conv.size(0), seg_conv.size(3), seg_conv.size(2), 1).cuda() # N * W * H * 1
        noise = (noise * self.noise_var).transpose(1, 3) # N * (channel_num) * H * W

        normalized = self.norm(seg_conv + noise) # 加噪声，并做归一化

        segmap = F.interpolate(seg_orig, size=seg_conv.size()[2:], mode='nearest') # 将原分割图直接下采样 N * 19 * 256 * 256 ===>> N * 19 * H * W

        if self.use_ST: # 加style_code，用SEAN
            batch, _, h, w = normalized.shape
            style_map = torch.zeros((batch, 512, h, w), device=normalized.device) # N * 512 * H * W

            for i in range(batch):
                for j in range(segmap.size(1)):
                    component_mask_area = torch.sum(segmap.byte()[i, j])
                    if component_mask_area > 0:

                        middle_mu = F.relu(self.per_style_convs[j](style_code[i][j])) # size: 512
                        component_mu = middle_mu.reshape(512, 1).expand(512, component_mask_area) # 512 * component_mask_area
                        style_map[i].masked_scatter_(segmap.byte()[i,j], component_mu) # 根据segmap的值，用component_mu 的值替换middle_avg[i]上的值
                        """
                        这里生成style_map的逻辑是：每个style_code（size: 512，共19个）先用全连接处理下得到size为512的middle_mu,
                        然后将middle_mu扩张成 512 * component_mask_area 的矩阵。
                        （.expand 方法将512个元素按列重复component_mask_area次，所以component_mu里面每一行的元素都是相同的）
                        最后同样的根据segmap上为1的位置，把component_mu上的值替换掉原style_map上的值。
                        这样就生成了新的style_map
                        """
            gamma_ST = self.conv_gamma(style_map)
            beta_ST = self.conv_beta(style_map)

            gamma_SPADE, beta_SPADE = self.spade(segmap)

            gamma_alpha = F.sigmoid(self.blending_gamma)
            beta_alpha = F.sigmoid(self.blending_beta)
            gamma_final = gamma_alpha * gamma_ST + (1 - gamma_alpha) * gamma_SPADE
            beta_final = beta_alpha * beta_ST + (1 - beta_alpha) * beta_SPADE

            out = normalized * (1 + gamma_final) + beta_final

        else: #不加style_code, 处理方式就跟spade一样
            gamma_SPADE, beta_SPADE = self.spade(segmap)
            out = normalized * (1 + gamma_SPADE) + beta_SPADE
        return out

class SPADE(nn.Module):
    def __init__(self, channel_num):
        super().__init__()

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(19, 128, kernel_size=3, padding=1),
            nn.ReLU()
        ) # 128: 中间维度，可自己设置

        self.mlp_gamma = nn.Conv2d(128, channel_num, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(128, channel_num, kernel_size=3, padding=1)

    def forward(self, segmap):

        inputmap = segmap

        actv = self.mlp_shared(inputmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return gamma, beta



# x = torch.randn((1, 3, 256, 256))
# seg= torch.randn((1, 19, 256, 256))
#
# m = StyleEncoder()
# m(x, seg)
