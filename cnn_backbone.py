# -*- coding:utf-8 -*-
# Author:Ding
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // ratio, in_planes, 1, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class ChannelAttention_1(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention_1, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes ** 2, in_planes ** 2 // ratio, 1, bias = False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes ** 2 // ratio, in_planes, 1, bias = False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        bi_out = torch.bmm(self.avg_pool(x).squeeze(-1), torch.transpose(self.max_pool(x).squeeze(-1), 1, 2)) / x.shape[
            1] ** 2
        bi_out = bi_out.view(-1, x.shape[1] ** 2).unsqueeze(-1).unsqueeze(-1)
        out = self.fc(bi_out)
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SpatialAttention_1(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention_1, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = kernel_size // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)


class AugmentedConv(nn.Module):
    def __init__(self, in_channels, kernel_size, dk, dv, Nh, shape = 0, relative = False, stride = 1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.qkv_conv1 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                   stride = stride, padding = self.padding)
        self.qkv_conv2 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                   stride = stride, padding = self.padding)
        self.qkv_conv12 = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size = self.kernel_size,
                                    stride = stride, padding = self.padding)

        self.attn_out1 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)
        self.attn_out2 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)
        self.attn_out12 = nn.Conv2d(self.dv, self.dv, kernel_size = 1, stride = 1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad = True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad = True))

    def forward(self, x1, x2, x12):
        # Input x
        # (batch_size, channels, height, width)
        batch, _, height, width = x1.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q1, flat_k1, flat_v1, q1, k1, v1 = self.compute_flat_qkv(x1, self.dk, self.dv, self.Nh, 1)
        flat_q2, flat_k2, flat_v2, q2, k2, v2 = self.compute_flat_qkv(x2, self.dk, self.dv, self.Nh, 2)
        flat_q12, flat_k12, flat_v12, q12, k12, v12 = self.compute_flat_qkv(x12, self.dk, self.dv, self.Nh, 12)
        logits1 = torch.matmul(flat_q1.transpose(2, 3), flat_k12)
        logits2 = torch.matmul(flat_q2.transpose(2, 3), flat_k12)
        logits12 = torch.matmul(flat_q12.transpose(2, 3), flat_k1) - torch.matmul(flat_q12.transpose(2, 3), flat_k2)

        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q1)
            logits1 += h_rel_logits
            logits1 += w_rel_logits
        weights1 = F.softmax(logits1, dim = -1)
        weights2 = F.softmax(logits2, dim = -1)
        weights12 = F.softmax(logits12, dim = -1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out1 = torch.matmul(weights1, flat_v1.transpose(2, 3))
        attn_out2 = torch.matmul(weights2, flat_v2.transpose(2, 3))
        attn_out12 = torch.matmul(weights12, flat_v12.transpose(2, 3))
        attn_out1 = torch.reshape(attn_out1, (batch, self.Nh, self.dv // self.Nh, height, width))
        attn_out2 = torch.reshape(attn_out2, (batch, self.Nh, self.dv // self.Nh, height, width))
        attn_out12 = torch.reshape(attn_out12, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out1 = self.combine_heads_2d(attn_out1)
        attn_out2 = self.combine_heads_2d(attn_out2)
        attn_out12 = self.combine_heads_2d(attn_out12)
        attn_out1 = self.attn_out1(attn_out1)
        attn_out2 = self.attn_out1(attn_out2)
        attn_out12 = self.attn_out1(attn_out12)
        return attn_out1, attn_out2, attn_out12

    def compute_flat_qkv(self, x, dk, dv, Nh, idx):
        if idx == 1:
            qkv = self.qkv_conv1(x)
        elif idx == 2:
            qkv = self.qkv_conv2(x)
        elif idx == 12:
            qkv = self.qkv_conv12(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim = 1)
        q = self.split_heads_2d(q, Nh)  # 将qkv按照head个数分割channel
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))  # 将空间维展开
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w")
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W))
        rel_logits = torch.unsqueeze(rel_logits, dim = 3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim = 3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x


# 定义一个基本卷积块，包括两个3x3的卷积层和一个BatchNorm层
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
            )

        self.ca = ChannelAttention_1(out_channels, ratio = 16)
        self.sa = SpatialAttention()

    def forward(self, x):
        identity = x
        # 第一层卷积和BN
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二层卷积和BN
        out = self.conv2(out)
        out = self.bn2(out)
        # CBAM
        out = self.ca(out) * out
        out = self.sa(out) * out
        # 如果输入的x与输出的out的尺寸不同，需要进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接
        out = self.relu(out)
        return out


# band-wise fusion
class BandWiseFusion(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        out_planes = in_planes // 2
        self.conv1 = nn.Conv2d(in_channels = in_planes, out_channels = out_planes, kernel_size = 3, padding = 1,
                               groups = out_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.gelu1 = nn.GELU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x1, x2):
        x = torch.zeros([x1.shape[0], 2 * x1.shape[1], x1.shape[2], x1.shape[3]]).to(x1.device)
        x[:, 0::2, ...] = x1
        x[:, 1::2, ...] = x2
        # TODO 是否需要在融合之前先进行依次逐通道的卷积
        out = self.dropout(self.gelu1(self.bn1(self.conv1(x))))

        return out


# 定义CNN骨干网络
class Backbone(nn.Module):
    def __init__(self, patch_size, band_size, dim):
        super(Backbone, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(band_size, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)
        self.conv12 = nn.Conv2d(band_size, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn12 = nn.BatchNorm2d(64)
        self.relu12 = nn.ReLU(inplace = True)
        # self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        # band fusion
        self.fuse_backbone = BandWiseFusion(2 * band_size)
        # 使用多个基本块构建网络
        # 孪生网络
        self.layer1 = self._make_layer(BasicBlock, 64, 1, stride = 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, stride = 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, stride = 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 1, stride = 1)
        # fuse网络
        self.in_channels = 64
        self.layer01 = self._make_layer(BasicBlock, 64, 1, stride = 1)
        self.layer02 = self._make_layer(BasicBlock, 64, 1, stride = 1)
        self.layer03 = self._make_layer(BasicBlock, 128, 1, stride = 1)
        self.layer04 = self._make_layer(BasicBlock, 128, 1, stride = 1)
        # cross attention
        self.cross_attn1 = AugmentedConv(in_channels = 64, kernel_size = 3, dk = 64, dv = 64, Nh = 4)
        self.cross_attn2 = AugmentedConv(in_channels = 64, kernel_size = 3, dk = 64, dv = 64, Nh = 4)
        self.cross_attn3 = AugmentedConv(in_channels = 128, kernel_size = 3, dk = 128, dv = 128, Nh = 4)
        self.cross_attn4 = AugmentedConv(in_channels = 128, kernel_size = 3, dk = 128, dv = 128, Nh = 4)
        # TODO 特征嵌入是每个scale用一个还是尽量用一个（分组卷积能克服通道数问题）
        # 构建特征嵌入模块
        self.feat_embed1 = nn.Conv2d(64, dim, kernel_size = patch_size, groups = 64)
        self.feat_embed2 = nn.Conv2d(64, dim, kernel_size = patch_size, groups = 64)
        self.feat_embed3 = nn.Conv2d(128, dim, kernel_size = patch_size, groups = 64)
        self.feat_embed4 = nn.Conv2d(128, dim, kernel_size = patch_size, groups = 64)

        # self.layer2 = self._make_layer(BasicBlock, 128, 2, stride = 2)
        # self.layer3 = self._make_layer(BasicBlock, 256, 2, stride = 2)
        # self.layer4 = self._make_layer(BasicBlock, 512, 2, stride = 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 第一个块的步幅为stride，后面的块步幅为1
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # fuse t1 & t2
        fuse_x = self.fuse_backbone(x1, x2)

        # 第一层卷积和BN
        out1 = self.relu(self.bn1(self.conv1(x1)))  # self.maxpool()
        out2 = self.relu(self.bn1(self.conv1(x2)))
        out12 = self.relu12(self.bn12(self.conv12(fuse_x)))

        out1 = self.layer1(out1)
        out2 = self.layer1(out2)
        out12 = self.layer01(out12)
        out1, out2, out12 = self.cross_attn1(out1, out2, out12)
        feat1 = self.feat_embed1(out12).squeeze()

        out1 = self.layer2(out1)
        out2 = self.layer2(out2)
        out12 = self.layer02(out12)
        out1, out2, out12 = self.cross_attn2(out1, out2, out12)
        feat2 = self.feat_embed2(out12).squeeze()

        out1 = self.layer3(out1)
        out2 = self.layer3(out2)
        out12 = self.layer03(out12)
        out1, out2, out12 = self.cross_attn3(out1, out2, out12)
        feat3 = self.feat_embed3(out12).squeeze()

        out1 = self.layer4(out1)
        out2 = self.layer4(out2)
        out12 = self.layer04(out12)
        out1, out2, out12 = self.cross_attn4(out1, out2, out12)
        feat4 = self.feat_embed4(out12).squeeze()

        return feat1.unsqueeze(1), feat2.unsqueeze(1), feat3.unsqueeze(1), feat4.unsqueeze(1)

# Example
# img1 = torch.rand([8, 154, 7, 7])
# img2 = torch.rand([8, 154, 7, 7])
# resnet = Backbone(154, 64)
# out1, out2, out3, out4 = resnet(img1, img2)
# print((out2.shape))
# seq = torch.cat((out1, out2, out3, out4), 1)
# img = torch.ones([8, 128, 7, 7])
# ca = ChannelAttention_1(in_planes = 128, ratio = 16)
# out = ca(img)
# print((out.shape))
