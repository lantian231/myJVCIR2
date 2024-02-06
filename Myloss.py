import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp
import math
from torchvision.models.vgg import vgg16
import numpy as np
import torchvision.models as models

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr - mg, 2)
        Drb = torch.pow(mr - mb, 2)
        Dgb = torch.pow(mb - mg, 2)
        k = torch.mean(Drg + Drb + Dgb)

        return k





class L_exp_two(nn.Module):

    def __init__(self,patch_size):
        super(L_exp_two, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
    def forward(self, x, high):

        x = torch.mean(x,1,keepdim=True)
        mean_low = self.pool(x)

        high = torch.mean(high, 1, keepdim=True)
        mean_high = self.pool(high)

        d = torch.mean(torch.pow(mean_high- mean_low,2))
        return d



class L_exp(nn.Module):

    def __init__(self, patch_size, mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x):
        b, c, h, w = x.shape
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight = 1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)

    def forward(self, x):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b, c, h, w = x.shape
        # x_de = x.cpu().detach().numpy()
        r, g, b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r - mr
        Dg = g - mg
        Db = b - mb
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Db, 2) + torch.pow(Dg, 2), 0.5)
        # print(k)

        k = torch.mean(k)
        return k


class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


# -------------------------------- 新增loss

class struct_loss(nn.Module):
    def __init__(self):
        super(struct_loss, self).__init__()

    def forward(self, low, high):
        loss = torch.mean(torch.square(high - low))
        return loss



class struct_loss_L1(nn.Module):
    def __init__(self):
        super(struct_loss_L1, self).__init__()

    def forward(self, low, high):
        loss = torch.mean(torch.abs(high - low))
        return loss





# 颜色loss，rgb角度最小
class color_loss(nn.Module):
    def __init__(self):
        super(color_loss, self).__init__()

    def forward(self, low, high):

        low_r, low_g, low_b = torch.split(low, 1, dim=1)
        high_r, high_g, high_b = torch.split(high, 1, dim=1)

        # 向量内积
        mul = low_r * high_r + low_g * high_g + low_b * high_b

        # 两个向量的绝对值
        low_value = torch.pow(low_r, 2) + torch.pow(low_g, 2) + torch.pow(low_b, 2)
        high_value = torch.pow(high_r, 2) + torch.pow(high_g, 2) + torch.pow(high_b, 2)

        loss = torch.mean(1 - torch.pow(mul, 2) / (low_value * high_value + 0.00001))

        return loss



# 平滑图像与groundTruth比较
class dx_dy_com(nn.Module):
    def __init__(self):
        super(dx_dy_com, self).__init__()

    def forward(self, low):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]

        pad_dx = F.pad(dx_low, (1, 0, 1, 0), 'constant', 0)

        dx_dy = pad_dx[:, :, 1:, 1:] - pad_dx[:, :, :-1, 1:]

        loss = torch.mean(torch.abs(dx_dy))

        return loss




# 平滑图像与groundTruth比较
class smooth_com(nn.Module):
    def __init__(self):
        super(smooth_com, self).__init__()

    def forward(self, low, high):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)
        pad_high = F.pad(high, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # high 梯度
        dx_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, 1:, :-1]
        dy_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, :-1, 1:]

        loss = torch.mean(torch.abs(dx_high - dx_low)) + torch.mean(torch.abs(dy_high - dy_low))

        return loss


#LL添加的loss:
class L_spa(nn.Module):           #空间一致性损失

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E












#LL的修改截止

# 平滑图像与groundTruth比较 L2
class smooth_com_L2(nn.Module):
    def __init__(self):
        super(smooth_com_L2, self).__init__()

    def forward(self, low, high):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)
        pad_high = F.pad(high, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # high 梯度
        dx_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, 1:, :-1]
        dy_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, :-1, 1:]

        loss = torch.mean(torch.square(dx_high - dx_low)) + torch.mean(torch.square(dy_high - dy_low))

        return loss


def dxy2(low):
    # 填充0
    pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)

    # low 梯度
    dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
    dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

    loss = torch.pow(torch.pow(dx_low, 2) + torch.pow(dy_low, 2), 0.5)

    return loss




class max_distance(nn.Module):
    def __init__(self):
        super(max_distance, self).__init__()
        self.pool = nn.AvgPool2d(4)

    def forward(self, low, high):

        avg_low = self.pool(low)
        avg_high = self.pool(high)
        loss = torch.mean(torch.abs(avg_high - avg_low))
        return loss






class L_spa_xy_4(nn.Module):

    def __init__(self):
        super(L_spa_xy_4, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        # self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        # self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)

        E = torch.mean(D_left + D_up)

        return E


class L_spa_xy_2(nn.Module):

    def __init__(self):
        super(L_spa_xy_2, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        # kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        # self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        # self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)

        E = torch.mean(D_left + D_up)

        return E



class L_spa_4(nn.Module):

    def __init__(self):
        super(L_spa_4, self).__init__()
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):

        org_pool = self.pool(org)
        enhance_pool = self.pool(enhance)

        loss = torch.mean(torch.square(enhance_pool - org_pool))

        return loss


class L_spa_2(nn.Module):

    def __init__(self):
        super(L_spa_2, self).__init__()
        self.pool = nn.AvgPool2d(2)

    def forward(self, org, enhance):
        org_pool = self.pool(org)
        enhance_pool = self.pool(enhance)

        loss = torch.mean(torch.square(enhance_pool - org_pool))

        return loss



# 平滑图像的梯度角度
class smooth_dxy2(nn.Module):
    def __init__(self):
        super(smooth_dxy2, self).__init__()

    def forward(self, low, high):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)
        pad_high = F.pad(high, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = 100 * (pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1])
        dy_low = 100 * (pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:])

        # high 梯度
        dx_high = 100 * (pad_high[:, :, 1:, 1:] - pad_high[:, :, 1:, :-1])
        dy_high = 100 * (pad_high[:, :, 1:, 1:] - pad_high[:, :, :-1, 1:])

        angle_divide = dx_low * dx_high + dy_low * dy_high
        angle_di = torch.pow((torch.square(dx_low) + torch.square(dy_low)), 0.5) * torch.pow((torch.square(dx_high) + torch.square(dy_high)), 0.5)

        loss = torch.mean(1 - angle_divide / (angle_di + 0.00001))

        return loss





# TV_Loss
class L_TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]



# 简单平滑损失
class smooth_low(nn.Module):
    def __init__(self):
        super(smooth_low, self).__init__()

    def forward(self, low):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        loss = torch.mean(torch.abs(dx_low)) + torch.mean(torch.abs(dy_low))

        return loss


# 简单平滑损失
class light_h(nn.Module):
    def __init__(self):
        super(light_h, self).__init__()

        self.pool = nn.MaxPool2d(4)

    def forward(self, light, high):
        r, g, b = torch.split(high, 1, 1)
        high = torch.max(r, g)
        high = torch.max(high, b)
        high = torch.cat([high, high, high], dim=1)

        loss = torch.mean(torch.abs(light - high))

        return loss



class LVD(nn.Module):
    def __init__(self):
        super(LVD, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=3)
        self.sample = nn.Upsample(scale_factor=3, mode='nearest')

    def forward(self, low):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # dx与均方差比值
        dx_pool = self.pool(dx_low)
        dx_up = self.sample(dx_pool)
        dx_up = F.interpolate(dx_up, (dx_low.size(2), dx_low.size(3)), mode='nearest')
        dx_lvd = torch.mean(torch.abs(dx_low) / (torch.abs(dx_up) + 0.01))

        # dx与均方差比值
        dy_pool = self.pool(dy_low)
        dy_up = self.sample(dy_pool)
        dy_up = F.interpolate(dy_up, (dy_low.size(2), dy_low.size(3)), mode='nearest')
        dy_lvd = torch.mean(torch.abs(dy_low) / (torch.abs(dy_up) + 0.01))

        loss = dx_lvd + dy_lvd

        return loss



class smooth_dxy(nn.Module):
    def __init__(self):
        super(smooth_dxy, self).__init__()
        self.pool = nn.MaxPool2d(4, stride=4)
        self.sample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, low, high):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)
        pad_high = F.pad(high, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # high 梯度
        dx_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, 1:, :-1]
        dy_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, :-1, 1:]

        dx_high_pool = self.pool(dx_high)
        dx_high_up = self.sample(dx_high_pool)
        dx_high_up = F.interpolate(dx_high_up, (dx_high.size(2), dx_high.size(3)), mode='nearest')

        dy_high_pool = self.pool(dy_high)
        dy_high_up = self.sample(dy_high_pool)
        dy_high_up = F.interpolate(dy_high_up, (dy_high.size(2), dy_high.size(3)), mode='nearest')

        loss1 = torch.abs(dx_low) / (torch.abs(dx_high_up) + 0.01)

        loss2 = torch.abs(dy_low) / (torch.abs(dy_high_up) + 0.01)

        loss = torch.mean(loss1 + loss2)

        return loss



# 约束梯度均方差，平滑图像
class LVD2(nn.Module):
    def __init__(self):
        super(LVD2, self).__init__()
        self.pool = nn.AvgPool2d(3, stride=3)
        self.sample = nn.Upsample(scale_factor=3, mode='nearest')

    def forward(self, low):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # dx与均方差比值
        dx_pool = self.pool(dx_low)
        dx_up = self.sample(dx_pool)
        dx_up = F.interpolate(dx_up, (dx_low.size(2), dx_low.size(3)), mode='nearest')
        dx_lvd = torch.mean(torch.abs(dx_up) * torch.abs(dx_low) / (torch.square(dx_low) + 0.01))

        # dx与均方差比值
        dy_pool = self.pool(dy_low)
        dy_up = self.sample(dy_pool)
        dy_up = F.interpolate(dy_up, (dy_low.size(2), dy_low.size(3)), mode='nearest')
        dy_lvd = torch.mean(torch.abs(dy_up) * torch.abs(dy_low) / (torch.square(dy_low) + 0.01))

        loss = dx_lvd + dy_lvd

        return loss




class smooth_light(nn.Module):
    def __init__(self):
        super(smooth_light, self).__init__()
        self.pool = nn.MaxPool2d(4, stride=4)
        self.sample = nn.Upsample(scale_factor=4, mode='nearest')

    def forward(self, low, high, enhance):

        # 填充0
        pad_low = F.pad(low, (1, 0, 1, 0), 'constant', 0)
        pad_high = F.pad(high, (1, 0, 1, 0), 'constant', 0)
        pad_enhance = F.pad(enhance, (1, 0, 1, 0), 'constant', 0)

        # low 梯度
        dx_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, 1:, :-1]
        dy_low = pad_low[:, :, 1:, 1:] - pad_low[:, :, :-1, 1:]

        # high 梯度
        dx_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, 1:, :-1]
        dy_high = pad_high[:, :, 1:, 1:] - pad_high[:, :, :-1, 1:]

        # high 梯度
        dx_enhance = pad_enhance[:, :, 1:, 1:] - pad_enhance[:, :, 1:, :-1]
        dy_enhance = pad_enhance[:, :, 1:, 1:] - pad_enhance[:, :, :-1, 1:]

        dlhx = (dx_low * high - dx_high * low) / (torch.pow(high, 2) + 0.01)
        loss1 = torch.mean(torch.abs(dlhx - dx_enhance))

        dlhy = (dy_low * high - dy_high * low) / (torch.pow(high, 2) + 0.01)
        loss2 = torch.mean(torch.abs(dlhy - dy_enhance))

        loss = loss1 + loss2

        return loss



class L2_light(nn.Module):
    def __init__(self):
        super(L2_light, self).__init__()

    def forward(self, low, high, enhance):

        lh = low / (torch.abs(high) + 0.01)
        loss = torch.mean(torch.square(lh - enhance))

        return loss





# gray loss
class gray_loss(nn.Module):
    def __init__(self):
        super(gray_loss, self).__init__()

    def forward(self, low):

        low_r, low_g, low_b = torch.split(low, 1, dim=1)

        lrg = torch.mean(torch.abs(low_r - low_g))
        lrb = torch.mean(torch.abs(low_r - low_b))
        lgb = torch.mean(torch.abs(low_g - low_b))

        loss = lrg + lrb + lgb

        return loss





# SSIM
# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)





def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret




def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window




def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


#VGG loss   我定义的vggloss
# 定义VGG网络和中间层
vgg = models.vgg16(pretrained=True).features.cuda()
for param in vgg.parameters():
    param.requires_grad_(False)

# 定义VGG loss函数
class vgg_loss(nn.Module):
    def __init__(self):
        super(vgg_loss, self).__init__()

    def forward(self, x, y):
        features_x = vgg(x)
        features_y = vgg(y)
        loss = torch.mean((features_x - features_y) ** 2)
        return loss


#Calculate color loss
class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, predict, target):
        b, c, h, w = target.shape
        target_view = target.view(b, c, h * w).permute(0, 2, 1)
        predict_view = predict.view(b, c, h * w).permute(0, 2, 1)
        target_norm = torch.nn.functional.normalize(target_view, dim=-1)
        predict_norm = torch.nn.functional.normalize(predict_view, dim=-1)
        cose_value = target_norm * predict_norm
        cose_value = torch.sum(cose_value, dim=-1)
        color_loss = torch.mean(1 - cose_value)

        return color_loss


class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss