import torch
import torch.nn as nn
import torch.nn.functional as F




def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

class TVLoss(nn.Module):                 #TV loss全称Total Variation Loss，其作用主要是降噪，图像中相邻像素值的差异可以通过降低TV Loss来一定程度上进行解决 ，从而保持图像的光滑性。
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
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