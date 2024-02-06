import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from model_simplify229_pre import model as pre_model
from torchvision.models.vgg import vgg16
#from night2day2 import BrightnessEnhancer
from torchvision import transforms

#from night2day4 import ImageBrightnessSampler
from night2day6guass import ImageBrightnessSampler
import numpy as np

from night2day_en2 import BrightnessCalculator
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr


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




class RFDB_enhance(nn.Module):
    def __init__(self):
        super(RFDB_enhance, self).__init__()
        self.pre_model = pre_model()
        self.vgg_model = vgg16(pretrained=True).features
        self.vgg_model.eval()
        self.image_enhance = ImageBrightnessSampler()             #en1
        #self.calculator = BrightnessCalculator("datasets/LOL/high_with_low")         #en2
        self.calculator_en1 = BrightnessCalculator("datasets/LOL/train_data/high")
        self.calculator_en2 = BrightnessCalculator("datasets/LOL/high_with_low")  # en2

        # self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        #为防止两个msrfe分支参数共享，需要对init里定义的一些功能分别定义：
        self.act = nn.PReLU()
        self.relu = nn.ReLU()

        channel = 32

        self.pool = conv_layer(channel, channel, 2, stride=2)

        self.pre1 = conv_layer(3, channel, 3)
        self.pre2 = conv_layer(channel, channel, 3)

        self.block1 = conv_block(channel, channel * 4)
        self.upconv1 = nn.ConvTranspose2d(channel * 4, channel * 2, kernel_size=2, stride=2)

        self.mid1 = conv_layer(channel * 2, channel * 2, 3)

        self.res_block1 = conv_block(channel * 2, channel * 2)

        self.block2 = conv_block(channel, channel * 2)
        self.upconv2 = nn.ConvTranspose2d(channel * 2, channel, kernel_size=2, stride=2)

        self.mid2 = conv_layer(channel, channel, 3)

        self.block3 = conv_block(channel, channel)

        self.post = conv_layer(32, 3, 3)

        self.enhance = conv_block(3, channel, True)
        #以上是第一分支的定义，以下是第二分支的定义，所有脚标加5
        self.act5 = nn.PReLU()
        self.relu5 = nn.ReLU()

        channel5 = 32

        self.pool5 = conv_layer(channel5, channel5, 2, stride=2)

        self.pre6 = conv_layer(3, channel5, 3)
        self.pre7 = conv_layer(channel5, channel5, 3)

        self.block6 = conv_block(channel5, channel5 * 4)
        self.upconv6 = nn.ConvTranspose2d(channel5 * 4, channel5 * 2, kernel_size=2, stride=2)

        self.mid6 = conv_layer(channel5 * 2, channel5 * 2, 3)

        self.res_block6 = conv_block(channel5 * 2, channel5 * 2)

        self.block7 = conv_block(channel5, channel5 * 2)
        self.upconv7 = nn.ConvTranspose2d(channel5 * 2, channel5, kernel_size=2, stride=2)

        self.mid7 = conv_layer(channel5, channel5, 3)

        self.block8 = conv_block(channel5, channel5)

        self.post5 = conv_layer(32, 3, 3)

        self.enhance5 = conv_block(3, channel5, True)
        #msrfe块第三分支，相对第二分支脚标加5
        self.act10 = nn.PReLU()
        self.relu10 = nn.ReLU()

        channel10 = 32

        self.pool10 = conv_layer(channel10, channel10, 2, stride=2)

        self.pre11 = conv_layer(3, channel10, 3)
        self.pre12 = conv_layer(channel10, channel10, 3)

        self.block11 = conv_block(channel10, channel10 * 4)
        self.upconv11 = nn.ConvTranspose2d(channel10 * 4, channel10 * 2, kernel_size=2, stride=2)

        self.mid11 = conv_layer(channel10 * 2, channel10 * 2, 3)

        self.res_block11 = conv_block(channel10 * 2, channel10 * 2)

        self.block12 = conv_block(channel10, channel10 * 2)
        self.upconv12 = nn.ConvTranspose2d(channel10 * 2, channel10, kernel_size=2, stride=2)

        self.mid12 = conv_layer(channel10, channel10, 3)

        self.block13 = conv_block(channel10, channel10)

        self.post10 = conv_layer(32, 3, 3)

        self.enhance10 = conv_block(3, channel5, True)
        #msrfe块第三分支结尾

        #atten块的内容   输入是3通道，输出是1通道
        # self.conv6 = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # self.conv7 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
        # self.conv8 = torch.nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2)
        # self.conv9 = torch.nn.Conv2d(16, 1, kernel_size=3, padding=1)
        #atten块的内容结尾

        #atten块改进：增加了归一化层、正则化技术、新的层次结构和激活函数、多尺度特征融合等，以提高模型的性能
        self.attconv6 = nn.Conv2d(3, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn6 = nn.BatchNorm2d(12)
        self.attdropout6 = nn.Dropout2d(p=0.2)

        self.attconv7 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn7 = nn.BatchNorm2d(12)
        self.attdropout7 = nn.Dropout2d(p=0.2)

        self.attconv8 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn8 = nn.BatchNorm2d(12)
        self.attdropout8 = nn.Dropout2d(p=0.2)

        self.attconv9 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn9 = nn.BatchNorm2d(12)
        self.attdropout9 = nn.Dropout2d(p=0.2)

        self.attconv10 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn10 = nn.BatchNorm2d(12)
        self.attdropout10 = nn.Dropout2d(p=0.2)

        self.attconv11 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn11 = nn.BatchNorm2d(12)
        self.attdropout11 = nn.Dropout2d(p=0.2)

        self.attconv12 = nn.Conv2d(12, 12, kernel_size=3, padding=2, dilation=2)
        self.attbn12 = nn.BatchNorm2d(12)
        self.attdropout12 = nn.Dropout2d(p=0.2)
        #atten块改进结尾

        #消除24.27中的上边分支FE块，将x,en1,en2转为12通道
        self.three_to_twelve_x = conv_layer(3, 12, 3)
        self.three_to_twelve_en1 = conv_layer(3, 12, 3)
        self.three_to_twelve_en2 = conv_layer(3, 12, 3)

    def forward(self, x):
        # x_h = 1 - x

        # 分支2预处理块MEG

        folder_path = 'datasets/LOL/train_data/high'
        # 从高光图像分布中抽取一个全局因子，增量图像
        d_en2 = self.calculator_en2.compute_brightness()

        # 将通道维度调整为第 1 维度，方便计算平均值
        # x2 = x.view(1, 3, -1)
        x2 = x.clone().detach().requires_grad_(True).view(1, 3, -1)
        # 计算每个通道的平均值
        channel_avg = torch.mean(x2, dim=2)
        # 将张量移动到CPU上并转换为NumPy数组
        # avg = channel_avg.cpu().numpy()
        avg = channel_avg.detach().cpu().numpy()
        # 提取数组中的三个数并计算平均值
        values = avg[0]  # 提取第一行的所有数值
        average = np.mean(values)
        x = x.clone().detach().requires_grad_(True)

        d_en1 = self.image_enhance.compute_brightness(folder_path)
        en1 = (d_en1 / average) * x
        en1_entropy = self.entropy(en1)
        # print("start_calu_entropy:")
        base_entropy = en1_entropy;
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for i in range(2):  # 选取拥有最高信息熵的张量作为en1
            d_en1_temp = self.image_enhance.compute_brightness(folder_path)
            # print("d_en1_temp:")
            # print(d_en1_temp)
            en1_temp = (d_en1_temp / average) * x
            en1_temp_entropy = self.entropy(en1_temp)
            # print("en1_temp_entropy:")
            # print(en1_temp_entropy)
            if (en1_temp_entropy > base_entropy):
                base_entropy = en1_temp_entropy
                # print("base_entropy:")
                # print(base_entropy)
                en1 = (d_en1_temp / average) * x
        # print("end_calu_entropy:")
        en2 = (d_en2 / average) * x

        #x = self.high_dynamic(en1, en2)
        # 分支2预处理块结尾
        #MSRFE块
        pre = self.act(self.pre1(x))
        pre = self.act(self.pre2(pre))

        fea2 = self.pool(pre)
        fea1 = self.pool(fea2)

        _, _, w1, h1 = fea2.size()
        fea1 = self.block1(fea1)
        fea1 = self.upconv1(fea1)
        fea1 = F.interpolate(fea1, (w1, h1), mode='nearest')
        fea1 = self.act(self.mid1(fea1))

        fea2 = self.block2(fea2)
        res1 = fea2 - fea1
        res1 = self.res_block1(res1)
        # res1 = self.upconv1(res1)

        _, _, w2, h2 = pre.size()
        add1 = fea2 - res1
        add1 = self.upconv2(add1)
        add1 = F.interpolate(add1, (w2, h2), mode='nearest')
        add1 = self.act(self.mid2(add1))

        fea3 = self.block3(pre)
        res2 = fea3 - add1
        res2 = self.block3(res2)
        # res2 = self.upconv2(res2)

        add2 = fea3 - res2
        add2 = self.act(self.post(add2))

        x_dill_0 = self.enhance(x)          #12通道  torch.Size([1,12,400,600])
        #MSRFE块结尾



        # MSRFE2块 变量与MSRFE块区分，指标加5，如pre->pre5
        pre5 = self.act5(self.pre6(en2))
        pre5 = self.act5(self.pre7(pre5))

        fea7 = self.pool5(pre5)
        fea6 = self.pool5(fea7)

        _, _, w6, h6 = fea7.size()
        fea6 = self.block6(fea6)
        fea6 = self.upconv6(fea6)
        fea6 = F.interpolate(fea6, (w6, h6), mode='nearest')      #
        fea6 = self.act5(self.mid6(fea6))

        fea7 = self.block7(fea7)
        res6 = fea7 - fea6
        res6 = self.res_block6(res6)
        # res1 = self.upconv1(res1)

        _, _, w7, h7 = pre5.size()
        add6 = fea7 - res6
        add6 = self.upconv7(add6)
        add6 = F.interpolate(add6, (w7, h7), mode='nearest')         #
        add6 = self.act5(self.mid7(add6))

        fea8 = self.block8(pre5)
        res7 = fea8 - add6
        res7 = self.block8(res7)
        # res2 = self.upconv2(res2)

        add7 = fea8 - res7
        add7 = self.act5(self.post5(add7))

        x_dill_0_5 = self.enhance5(x)            #为与x_dill_0区分，加了_5，变成x_dill_0_5，12通道的
        # MSRFE2块结尾

        # MSRFE3块 变量与MSRFE2块区分，指标加5，如pre5->pre10
        pre10 = self.act10(self.pre11(en1))
        pre10 = self.act10(self.pre12(pre10))

        fea12 = self.pool10(pre10)
        fea11 = self.pool10(fea12)

        _, _, w11, h11 = fea12.size()
        fea11 = self.block11(fea11)
        fea11 = self.upconv11(fea11)
        fea11 = F.interpolate(fea11, (w11, h11), mode='nearest')      #
        fea11 = self.act10(self.mid11(fea11))

        fea12 = self.block12(fea12)
        res11 = fea12 - fea11
        res11 = self.res_block11(res11)
        # res1 = self.upconv1(res1)

        _, _, w12, h12 = pre10.size()
        add11 = fea12 - res11
        add11 = self.upconv12(add11)
        add11 = F.interpolate(add11, (w12, h12), mode='nearest')         #
        add11 = self.act10(self.mid12(add11))

        fea13 = self.block13(pre10)
        res12 = fea13 - add11
        res12 = self.block13(res12)
        # res2 = self.upconv2(res2)

        add12 = fea13 - res12
        add12 = self.act10(self.post10(add12))

        x_dill_0_10 = self.enhance10(x)            #为与x_dill_5区分，加了_5，变成x_dill_0_10，12通道的
        # MSRFE3块结尾

        #TBEFN的atten块

        # atten = F.relu(self.conv6(x))
        # atten = F.relu(self.conv7(atten))
        # atten = F.relu(self.conv8(atten))
        # atten = self.conv9(atten)
        ##atten =  F.sigmoid(atten)
        #atten改进部分，替换上边四行atten
        atten1 = F.relu(self.attbn6(self.attconv6(x)))
        atten1 = self.attdropout6(atten1)

        atten2 = F.relu(self.attbn7(self.attconv7(atten1)))
        atten2 = self.attdropout7(atten2)

        atten3 = F.relu(self.attbn8(self.attconv8(atten2)))
        atten3 = self.attdropout8(atten3)

        atten4 = F.relu(self.attbn9(self.attconv9(atten3)))
        atten4 = self.attdropout9(atten4)

        atten5 = F.relu(self.attbn10(self.attconv10(atten4)))
        atten5 = self.attdropout10(atten5)

        atten6 = F.relu(self.attbn11(self.attconv11(atten5)))
        atten6 = self.attdropout11(atten6)

        atten7 = F.relu(self.attbn12(self.attconv12(atten6)))
        atten7 = self.attdropout12(atten7)

        # 多尺度特征融合
        atten = atten1 + atten2 + atten3 + atten4 + atten5 + atten6 + atten7

        # atten = torch.mean(atten, dim=1)             #这两行代码让atten变成单通道的
        # atten = atten.unsqueeze(1)
        #atten改进结尾
        atten = F.relu(atten) - F.relu(atten - 1.0)               #torch.Size([1, 12, 400, 600])
        # x2 = x.repeat(1, 4, 1, 1)                                        #三通转12通
        # enhance1 = en1.repeat(1, 4, 1, 1)
        # enhance2 = en2.repeat(1, 4, 1, 1)
        # #
        # x_dill_0 = F.relu(x2 * x_dill_0) - F.relu(x2 * x_dill_0 - 1.0)                          #第一分支
        # x_dill_0_5 = F.relu(enhance2 * x_dill_0_5) - F.relu(enhance2 * x_dill_0_5 - 1.0)        #第二分支
        # x_dill_0_10 = F.relu(enhance1 * x_dill_0_10) - F.relu(enhance1 * x_dill_0_10 - 1.0)     #第三分支

        #尝试跳过FE块，即删除FE块，详情见F:\lilong\研究生的活\苏老师任务\卫师兄项目\low light\模型性能指标相关\30.仔细研究MSRFE块时发现问题，关于x_dill_0\发现问题前，性能达到24.27的网络结构图以及备份\性能改进\7.在6的基础上，师兄电脑，尝试消除FE块
        xq = self.three_to_twelve_x(x)
        en1q = self.three_to_twelve_en1(en1)
        en2q = self.three_to_twelve_en2(en2)
        #色偏诊断        没用
        # x_dill_0 = F.relu(x2 * x_dill_0) - F.relu(x2 * x_dill_0 - 1.0)                          #第一分支
        # en2q = F.relu(en2q*xq) - F.relu(en2q*xq - 1.0)        #第二分支
        # en1q = F.relu(en1q*xq) - F.relu(en1q*xq - 1.0)        #第三分支

        quality_score_0 = self.evaluate_quality(xq)                   #计算每个分支的质量分数
        quality_score_5 = self.evaluate_quality(en2q)
        quality_score_10 = self.evaluate_quality(en1q)

        #假设产生的质量评分范围是-1到1，则将其转换为0-1之间
        quality_score_0 = (quality_score_0+1)/2
        quality_score_5 = (quality_score_5 + 1) / 2
        quality_score_10 = (quality_score_10 + 1) / 2

        if quality_score_0 == 0:
            quality_score_0 += 0.01
        if quality_score_5 == 0:
            quality_score_5 += 0.01
        if quality_score_10 == 0:
            quality_score_10 += 0.01

        fusion_weight1 = quality_score_0/(quality_score_0+quality_score_5)   #计算一二分支融合的权重值
        atten1 = fusion_weight1*atten * xq + (1-fusion_weight1*atten) * en2q

        quality_score_atten1 = self.evaluate_quality(atten1)                 #计算一二分支融合后的结果的质量分数
        quality_score_atten1 = (quality_score_atten1 + 1) / 2
        if quality_score_atten1 == 0:
            quality_score_atten1 += 0.01
        fusion_weight2 = quality_score_atten1 / (quality_score_atten1 + quality_score_10)      #计算（一二）与第三分支融合的权重值
        atten2 = fusion_weight2*atten * atten1 + (1 - fusion_weight2*atten) * en1q
        #TBEFN的atten块结尾
        #打印两个MSRFE和注意力模块之间的中间结果
        t1, t2, t3, t4 = torch.split(x_dill_0, 3, dim=1)
        t5, t6, t7, t8 = torch.split(x_dill_0_5, 3, dim=1)
        #曲线调整块
        #print(x_dill_0)
        r1, r2, r3, r4 = torch.split(atten2, 3, dim=1)         #衔接我加的atten块，将x_dill_0改为atten_result，都是12通道

        #色偏诊断
        #add12 = F.relu(add12) - F.relu(add12 - 1.0)         没用
        #对add2,add7,add12进行权重自适应计算
        add2_score = self.evaluate_quality(add2)  # 计算每个分支的质量分数
        add7_score = self.evaluate_quality(add7)
        add12_score = self.evaluate_quality(add12)

        # 假设产生的质量评分范围是-1到1，则将其转换为0-1之间
        add2_score = (add2_score + 1) / 2
        add7_score = (add7_score + 1) / 2
        add12_score = (add12_score + 1) / 2

        if add2_score == 0:
            add2_score += 0.01
        if add7_score == 0:
            add7_score += 0.01
        if add12_score == 0:
            add12_score += 0.01

        add2_weight = add2_score / (add2_score + add7_score + add12_score)
        add7_weight = add7_score / (add2_score + add7_score + add12_score)
        add12_weight = add12_score / (add2_score + add7_score + add12_score)

        light = add2_weight*(add2 + r1 * (1 / math.pi) * torch.sin(math.pi * add2)) + add7_weight*(add7 + r1 * (1 / math.pi) * torch.sin(math.pi * add7)) + add12_weight*(add12 + r1 * (1 / math.pi) * torch.sin(math.pi * add12))
        light = light + r2 * (1 / math.pi) * torch.sin(math.pi * light)
        light = light + r3 * (1 / math.pi) * torch.sin(math.pi * light)
        light = light + r4 * (1 / math.pi) * torch.sin(math.pi * light)

        out_fea = torch.cat([r1, r2, r3, r4], dim=1)
        #曲线调整块结尾
        #add2是平滑图像，light是增强后的图像
        #剑走偏锋，哪个指标高，就返回哪一个
        # pre_score = self.evaluate_quality(light)
        # pre_score = (pre_score + 1) / 2
        # if pre_score == 0:
        #     pre_score += 0.01
        #
        # max_score = self.find_max(add2_score,add7_score,add12_score,pre_score)
        # if(add2_score == max_score):            #如果add2性能最优
        #     return out_fea, light, add2, add7 ,add12,en1,en2         #add2作为返回的pre
        # if (add7_score == max_score):  # 如果add7性能最优
        #     return out_fea, add2, add7, light, add12, en1, en2  # add7作为返回的pre
        # if (add7_score == max_score):  # 如果add12性能最优
        #     return out_fea, add2, add12, add7, light, en1, en2

        #return out_fea, add2, light, add7,add12
        return out_fea, add2, light, add7 ,add12,en1,en2            #最原始的返回顺序

    def evaluate_quality(self, input_tensor,flag=0):
        #print(input_tensor.size())      #torch.Size([16, 3, 128, 128])
        num_channels = input_tensor.shape[1]  # 获取输入张量的通道数

        if num_channels == 12:
            # 定义通道合并权重
            channel_weights = torch.tensor([0.2989, 0.587, 0.114, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(
                1, 12, 1, 1)

            # 将输入张量和权重都移动到相同的设备上
            input_tensor = input_tensor.to(channel_weights.device)

            # 将输入张量与权重进行点乘，实现通道合并
            converted_tensor = torch.sum(input_tensor * channel_weights, dim=1, keepdim=True)
        else:
            converted_tensor = input_tensor  # 若通道数不为12，则直接返回输入张量

        #将转换后的张量移动到与VGG模型参数相同的设备上
        converted_tensor = converted_tensor.to(next(self.vgg_model.parameters()).device)

        # 调整输入张量的通道数
        converted_tensor = converted_tensor.expand(-1, 3, -1, -1)

        # 获取模型的特征提取器部分
        feature_extractor = self.vgg_model


        """
        feature_extractor
        0
        torch.Size([16, 64, 128, 128])               0
        torch.Size([16, 64, 128, 128])
        torch.Size([16, 64, 128, 128])
        torch.Size([16, 64, 128, 128])               3
        torch.Size([16, 64, 64, 64])
        torch.Size([16, 128, 64, 64])                5
        torch.Size([16, 128, 64, 64])
        torch.Size([16, 128, 64, 64])
        torch.Size([16, 128, 64, 64])              8
        torch.Size([16, 128, 32, 32])               9
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 256, 16, 16])              16
        torch.Size([16, 512, 16, 16])              
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 8, 8])                23
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 8, 8])
        torch.Size([16, 512, 4, 4])               30
        1
        """
        # 存储特征图的列表
        feature_maps = []
        value_sum = 0
        #print(0)
        # 通过选择的层传递输入图像以获取特征图
        '''
        选取到的层
        0
        torch.Size([16, 64, 128, 128])
        torch.Size([16, 128, 64, 64])
        torch.Size([16, 256, 32, 32])
        torch.Size([16, 512, 16, 16])
        torch.Size([16, 512, 8, 8])
        1
        '''
        input_size = input_tensor.size()[2]
        #print(input_size)
        # 选择需要获取特征图的层的索引（从0开始）可选0,3,6,10,14，数字越小，对细节越关注，数字越大，对上下文越关注
        selected_layers = [3, 9, 16, 23, 30]
        #print(0)
        count=0    #用来记录循环到了第几层
        converted_tensor = converted_tensor.clone()
        for idx, layer in enumerate(feature_extractor):
            converted_tensor = layer(converted_tensor)
            #print(converted_tensor.size())
            if idx in selected_layers:
                #print(converted_tensor.size())
                #count += 1
                if flag == 2 and count == 0 and input_size == 128:      #如果输入的是add2,即暗光分支,直接返回vgg第0层作为权值矩阵进行加权
                    # 使用卷积来将其转换为目标大小[16, 3, 128, 128]
                    conv_layer = nn.Conv2d(64, 3, kernel_size=1).to('cuda')  # 1x1卷积层
                    converted_tensor = conv_layer(converted_tensor)
                    # 使用双线性上采样将输出张量的大小调整为目标大小[16, 3, 128, 128]
                    output_tensor = F.interpolate(converted_tensor, size=(128, 128), mode='bilinear', align_corners=False)

                    return output_tensor
                if flag == 7 and count == 2 and input_size == 128:      #如果输入的是add7,即中等分支,返回vgg中间层作为权值矩阵进行加权
                    # 使用卷积来将其转换为目标大小[16, 3, 128, 128]
                    conv_layer = nn.Conv2d(256, 3, kernel_size=1).to('cuda')  # 1x1卷积层
                    converted_tensor = conv_layer(converted_tensor)
                    # 使用双线性上采样将输出张量的大小调整为目标大小[16, 3, 128, 128]
                    output_tensor = F.interpolate(converted_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                    return output_tensor
                if flag == 12 and count == 4 and input_size == 128:      #如果输入的是add12,即亮光分支,返回vgg第14层作为权值矩阵进行加权
                    # 使用卷积来将其转换为目标大小[16, 3, 128, 128]
                    conv_layer = nn.Conv2d(512, 3, kernel_size=1).to('cuda')  # 1x1卷积层
                    converted_tensor = conv_layer(converted_tensor)
                    # 使用双线性上采样将输出张量的大小调整为目标大小[16, 3, 128, 128]
                    output_tensor = F.interpolate(converted_tensor, size=(128, 128), mode='bilinear', align_corners=False)
                    return output_tensor
                feature_maps.append(converted_tensor.clone())
                #print(converted_tensor.size())
                value_sum += torch.mean(converted_tensor)
                count += 1
        #print(1)
        # 计算特征图的平均值
        #average_feature_map = torch.mean(torch.cat(feature_maps, dim=0), dim=0)

        return value_sum/5.0


    def find_max(self,a, b, c, d):
        max_num = a
        if b > max_num:
            max_num = b
        if c > max_num:
            max_num = c
        if d > max_num:
            max_num = d
        return max_num

    def entropy(self,input_tensor):
        # 使用softmax函数将输入张量转换为概率分布
        prob_dist = F.softmax(input_tensor, dim=1)

        # 计算信息熵
        entropy = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-10), dim=1)
        avg_entropy = torch.mean(entropy)
        return avg_entropy

    def high_dynamic(self,x1,x2):
        # 计算 x1 和 x2 的平均值 a1 和 a2
        a1 = x1.mean()
        a2 = x2.mean()
        x1 = x1.clone().to("cuda")
        x2 = x2.clone().to("cuda")
        # 计算 a 的平均值
        a = (a1 + a2) / 2

        # 根据 a 进行阈值处理，保留 x1 中小于 a 的像素并置零其他像素
        y1 = torch.where(x1 < a, 1.2*x1, torch.tensor(0.0, dtype=torch.float32).to("cuda"))

        # 根据 a 进行阈值处理，保留 x2 中大于 a 的像素并置零其他像素
        y2 = torch.where(x2 > a, x2, torch.tensor(0.0, dtype=torch.float32).to("cuda"))
        y = y1 + y2
        # 对 y1 和 y2 进行加和
        # 使用掩码将 y 中的0值替换为 x1 中的对应像素值
        mask = (y == 0)
        x = (x1+x2)/2.0
        y[mask] = x[mask]
        return y

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.feature = RFDB_enhance()

    def forward(self, x):

        # 加载预训练好的子网络模型
        pre = pre_model()
        pre.load_state_dict(torch.load('models_PRE_simplify229_smooth3\SGN_DLN_pre_100.pth'))
        # 将预训练的子网络模型的参数复制给主网络的子模块
        self.feature.pre_model.load_state_dict(pre.state_dict())
        RFDB = self.feature(x)
        return RFDB



