import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class FusionLayer_y(nn.Module):
    def __init__(self, inchannel, outchannel, reduction=16):
        super(FusionLayer_y, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(inchannel // reduction, inchannel, bias=False),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(inchannel, outchannel, 1, 1, 0, bias=True)
        self.relu = nn.PReLU()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        y = y + x
        y = self.relu(self.conv(y))
        return y





def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation          #逆卷积
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)





class conv_block(nn.Module):
    def __init__(self, inChannel, outChannel, flag=False):
        super(conv_block, self).__init__()

        self.act = nn.PReLU()

        reduction = 16
        if outChannel == 32:
            reduction = 8

        if outChannel == 3:
            reduction = 3

        outChannel2 = outChannel
        if flag:
            outChannel2 = 12



        self.conv_h = conv_layer(inChannel, outChannel, 3)
        self.dill_conv1 = conv_layer(outChannel, outChannel, 3)
        self.dill_conv2 = conv_layer(outChannel, outChannel, 3)
        self.feature = FusionLayer_y(outChannel, outChannel2, reduction)

    def forward(self, x):

        x = self.act(self.conv_h(x))
        x = self.act(self.dill_conv1(x) + x)
        x = self.act(self.dill_conv2(x) + x)
        x = self.feature(x)

        return x




# class RFDB_enhance(nn.Module):
#     def __init__(self):
#         super(RFDB_enhance, self).__init__()
#
#
#
#
#     def forward(self, x):
#
#         return out_fea, add2, light, add7
#         #return out_fea, add2, light, add7 ,conv15_result,t1, t2, t3, t4,t5, t6, t7, t8,r1, r2, r3, r4
class ExposureNet(nn.Module):
    def __init__(self):
        super(ExposureNet, self).__init__()

        self.conv11 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv14 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv15 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.out_acti = nn.ReLU()
        self.enhance = conv_block(3,3)

        #atten块
        self.conv6 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv7 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
        self.conv8 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
        self.conv9 = torch.nn.Conv2d(16, 3, kernel_size=3, padding=1)


    def forward(self, x):
        atten = F.relu(self.conv6(x))
        atten = F.relu(self.conv7(atten))
        atten = F.relu(self.conv8(atten))
        atten = self.conv9(atten)

        conv11_out = self.conv11(x)

        conv12_in = conv11_out
        conv12_out = self.conv12(conv12_in)

        conv13_in = conv11_out + conv12_out
        conv13_out = self.conv13(conv13_in)

        conv14_in = conv13_in + conv13_out
        conv14_out = self.conv14(conv14_in)

        conv15_in = conv14_in + conv14_out
        conv15_out = self.conv15(conv15_in)

        x_e = self.enhance(x)

        conv15_result = F.relu(x + conv15_out) - F.relu(x + conv15_out - 1.0)
        conv15_result = F.relu(x + conv15_result)
        # 生成两种不同曝光程度的图像
        enhanced_image = atten * x_e + (1-atten) * conv15_result

        return enhanced_image


class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        #self.feature = RFDB_enhance()
        self.feature = ExposureNet()
    def forward(self, x):

        return self.feature(x)



