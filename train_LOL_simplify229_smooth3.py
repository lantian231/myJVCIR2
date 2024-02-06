import argparse
import itertools
import math
import os
import time
from os import listdir
from os.path import join
import shutil

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from skimage.metrics._structural_similarity import structural_similarity as compare_ssim
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from torch.utils.data import DataLoader

import lib.pytorch_ssim as pytorch_ssim
from lib.data import get_training_set, is_image_file, get_Low_light_training_set
from lib.utils import TVLoss, print_network
import Myloss

from model_simplify229 import model

Name_Exp = 'SGN_DLN_new'
exp = Experiment(Name_Exp)
# exp.observers.append(MongoObserver(url='Host:27017', db_name='low_light'))
exp.add_source_file("train_LOL_simplify229_smooth3.py")
exp.add_source_file("model_simplify229.py")
exp.add_source_file("lib/dataset.py")
exp.captured_out_filter = apply_backspaces_and_linefeeds


@exp.config
def cfg():
    parser = argparse.ArgumentParser(description='PyTorch Low-Light Enhancement')
    parser.add_argument('--batchSize', type=int, default=16, help='training batch size')#32改16，终于不报内存不够了
    parser.add_argument('--nEpochs', type=int, default=3000, help='number of epochs to train for')
    parser.add_argument('--snapshots', type=int, default=10, help='Snapshots')
    parser.add_argument('--start_iter', type=int, default=0, help='Starting Epoch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
    parser.add_argument('--gpu_mode', type=bool, default=True)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')#8改1
    parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped LR image')
    parser.add_argument('--save_folder', default='models_LOL_simplify229_smooth3/', help='Location to save checkpoint models')
    parser.add_argument('--isdimColor', default=False, help='synthesis at HSV color space')
    parser.add_argument('--isaddNoise', default=False, help='synthesis with noise')
    opt = parser.parse_args()


def checkpoint(model, epoch, opt):
    try:
        os.stat(opt.save_folder)
    except:
        os.mkdir(opt.save_folder)

    model_out_path = opt.save_folder + "{}_{}.pth".format(Name_Exp, epoch)
    torch.save(model.state_dict(), model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    return model_out_path


def log_metrics(_run, logs, iter, end_str=" "):
    str_print = ''
    for key, value in logs.items():
        _run.log_scalar(key, float(value), iter)
        str_print = str_print + "%s: %.4f || " % (key, value)
    print(str_print, end=end_str)


def eval(model, epoch):
    print("==> Start testing")
    tStart = time.time()
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    model.eval()
    test_LL_folder = "datasets/LOL/eval15/low/"
    test_NL_folder = "datasets/LOL/eval15/high/"
    test_est_folder = "outputs_LOL_simplify229_smooth3/eopch_%04d/" % (epoch)
    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            LL = trans(Image.open(test_LL_list[i]).convert('RGB')).unsqueeze(0).cuda()
            _, _, prediction ,_,_,_,_= model(LL)                #改了双分支smoothloss后这里报错，too many values to unpack (expected 3)，我在后边加了一个,_
            prediction = prediction[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    shutil.rmtree(test_est_folder)
    return psnr_score, ssim_score

def big_eval(model, epoch):
    print("==> Start testing")
    tStart = time.time()
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    model.eval()
    test_LL_folder = "datasets/LOL/eval15/low/"
    test_NL_folder = "datasets/LOL/eval15/high/"
    test_est_folder = "outputs_LOL_simplify229_smooth3/eopch_%04d/" % (epoch)
    try:
        os.stat(test_est_folder)
    except:
        os.makedirs(test_est_folder)

    test_LL_list = [join(test_LL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    test_NL_list = [join(test_NL_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    est_list = [join(test_est_folder, x) for x in sorted(listdir(test_LL_folder)) if is_image_file(x)]
    for i in range(test_LL_list.__len__()):
        with torch.no_grad():
            temp_gt = cv2.imread(test_NL_list[i])
            LL = trans(Image.open(test_LL_list[i]).convert('RGB')).unsqueeze(0).cuda()
            _, _, prediction, _, _, _, _ = model(LL)
            prediction = prediction[0].cpu().numpy().transpose(channel_swap)
            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            temp_psnr_val = compare_psnr(temp_gt, prediction, data_range=255)
            for j in range(3):                                      #擂台法,得到同一张弱光图像的多个预测结果，选一个最好的
                _, _, temp_prediction ,_,_,_,_= model(LL)                #改了双分支smoothloss后这里报错，too many values to unpack (expected 3)，我在后边加了一个,_
                temp_prediction = temp_prediction[0].cpu().numpy().transpose(channel_swap)
                temp_prediction = temp_prediction * 255.0
                temp_prediction = temp_prediction.clip(0, 255)
                challenger_psnr_val = compare_psnr(temp_gt, temp_prediction, data_range=255)
                if challenger_psnr_val > temp_psnr_val:
                    prediction = temp_prediction
                    temp_psnr_val = challenger_psnr_val
            Image.fromarray(np.uint8(prediction)).save(est_list[i])
    psnr_score = 0.0
    ssim_score = 0.0
    for i in range(test_NL_list.__len__()):
        gt = cv2.imread(test_NL_list[i])
        est = cv2.imread(est_list[i])
        psnr_val = compare_psnr(gt, est, data_range=255)
        ssim_val = compare_ssim(gt, est, multichannel=True)
        psnr_score = psnr_score + psnr_val
        ssim_score = ssim_score + ssim_val
    psnr_score = psnr_score / (test_NL_list.__len__())
    ssim_score = ssim_score / (test_NL_list.__len__())
    print("time: {:.2f} seconds ==> ".format(time.time() - tStart), end=" ")
    shutil.rmtree(test_est_folder)
    return psnr_score, ssim_score

@exp.automain
def main(opt, _run):
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)
        cudnn.benchmark = True
    gpus_list = range(opt.gpus)

    # =============================#
    #   Prepare training data     #
    # =============================#
    # first use the synthesis data (from VOC 2007) to train the model, then use the LOL real data to fine tune
    print('===> Prepare training data')
    train_set = get_training_set('datasets/LOL/train_data', upscale_factor=1, patch_size=opt.patch_size,
                                 data_augmentation=True)
    # train_set = get_training_set("datasets/LOL/train", 1, opt.patch_size, True) # uncomment it to do the fine tuning
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      pin_memory=True, shuffle=True, drop_last=True)
    # =============================#
    #          Build model        #
    # =============================#
    print('===> Build model')
    lighten = model()
    # lighten = torch.nn.DataParallel(lighten)
    # lighten.load_state_dict(torch.load("models_VOC_new/SGN_DLN_500.pth", map_location=lambda storage, loc: storage), strict=True)

    print('---------- Networks architecture -------------')
    print_network(lighten)

    print('----------------------------------------------')
    if cuda:
        lighten = lighten.cuda()

    # =============================#
    #         Loss function       #
    # =============================#
    L1_criterion = nn.L1Loss()               #亮度损失
    L1_char_loss = Myloss.L1_Charbonnier_loss()  # 改进l1loss为charbonnierloss
    smooth = Myloss.smooth_com()             # 平滑图像与groundTruth比较
    TV_loss = TVLoss()                       #TV loss全称Total Variation Loss，其作用主要是降噪，图像中相邻像素值的差异可以通过降低TV Loss来一定程度上进行解决 ，从而保持图像的光滑性
    mse_loss = torch.nn.MSELoss()            #1、均方误差（L2损失）均方误差(MSE)是最常用的回归损失函数，计算方法是求预测值与真实值之间距离的平方和，
    ssim = pytorch_ssim.SSIM()               #结构相似性损失
    vgg_loss = Myloss.vgg_loss()             #1.我加的vggloss
    color_loss = Myloss.color_loss()  # 1.我加的颜色损失

    #lspa = Myloss.L_spa
    if cuda:
        gpus_list = range(opt.gpus)
        mse_loss = mse_loss.cuda()
        L1_criterion = L1_criterion.cuda()
        smooth = smooth.cuda()
        TV_loss = TV_loss.cuda()
        ssim = ssim.cuda(gpus_list[0])
        vgg_loss = vgg_loss.cuda()           #2.我加的VGG loss
        L1_char_loss = L1_char_loss.cuda()
        color_loss = color_loss.cuda()  # 2.我加的颜色损失c

    # =============================#
    #         Optimizer            #
    # =============================#
    parameters = [lighten.parameters()]
    optimizer = optim.Adam(itertools.chain(*parameters), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

    # =============================#
    #         Training             #
    # =============================#
    psnr_score, ssim_score = eval(lighten, 0)
    print(psnr_score)
    max_psnr = psnr_score               #在训练时记录并打印最大的psnr,ssim的值
    max_ssim = ssim_score
    max_psnr_epoch = 0
    max_ssim_epoch = 0
    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        print('===> training epoch %d' % epoch)
        epoch_loss = 0
        lighten.train()

        tStart_epoch = time.time()
        for iteration, batch in enumerate(training_data_loader, 1):
            over_Iter = epoch * len(training_data_loader) + iteration
            optimizer.zero_grad()

            LL_t, NL_t = batch[0], batch[1]
            if cuda:
                LL_t = LL_t.cuda()
                NL_t = NL_t.cuda()

            t0 = time.time()

            _, add2, pred_t, add7 ,add12,_,_= lighten(LL_t)               #输入弱光图像

            ssim_loss = (1 - ssim(pred_t, NL_t)) + 0.5*(1 - ssim(add2, NL_t)) + 0.5*(1 - ssim(add12, NL_t)) + 0.5*(1 - ssim(add7, NL_t))            #猜测：pred_t是增强后的图像，NL_t是GT，而LL是low light，是输入的弱光图像
            tv_loss = L1_criterion(pred_t, NL_t) + 0.5*L1_criterion(add2, NL_t) + 0.5*L1_criterion(add7, NL_t) + 0.5*L1_criterion(add12, NL_t)        #NL_t是GT
            #smooth_loss = 2 * smooth(add2, NL_t)          #add2是平滑图像
            smooth_loss = 2.0*smooth(add2, NL_t) + 2.0*smooth(add7, NL_t) + 2.0* smooth(add12, NL_t)      #我改-的，add2是第一分支，add7是第分支的
            vgg = vgg_loss(NL_t,pred_t) + 0.5*vgg_loss(NL_t,add2) + 0.5*vgg_loss(NL_t,add7) + 0.5*vgg_loss(NL_t,add12)              #3.我加的VGGloss
            L1_char = L1_char_loss(pred_t, NL_t) + 0.5 * L1_char_loss(add2, NL_t) + 0.5 * L1_char_loss(add7, NL_t) + 0.5 * L1_char_loss(add12, NL_t)
            color = color_loss(pred_t,NL_t)
            loss = ssim_loss + 0.4*tv_loss + smooth_loss + 0.125*vgg + L1_char + 0.1*color

            loss.backward()
            optimizer.step()
            t1 = time.time()

            epoch_loss += loss

            if iteration % 10 == 0:
                print("Epoch: %d/%d || Iter: %d/%d " % (epoch, opt.nEpochs, iteration, len(training_data_loader)),
                      end=" ==> ")
                logs = {
                    "loss": loss.data,
                    "ssim_loss": ssim_loss.data,
                    "tv_loss": tv_loss.data,
                    "smooth_loss": smooth_loss.data,
                    #"smooth_loss2": smooth_loss2.data,
                    "vgg": vgg.data,
                    "L1_char_loss": L1_char.data,
                    "color_loss": color.data
                }
                log_metrics(_run, logs, over_Iter)
                print("time: {:.4f} s".format(t1 - t0))

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}; ==> {:.2f} seconds".format(epoch, epoch_loss / len(
            training_data_loader), time.time() - tStart_epoch))
        _run.log_scalar("epoch_loss", float(epoch_loss / len(training_data_loader)), epoch)

        if epoch % (opt.snapshots) == 0:
            file_checkpoint = checkpoint(lighten, epoch, opt)
            exp.add_artifact(file_checkpoint)

            psnr_score, ssim_score = big_eval(lighten, epoch)
            if (psnr_score > max_psnr):
                max_psnr = psnr_score
                max_psnr_epoch = epoch
            if (ssim_score > max_ssim):
                max_ssim = ssim_score
                max_ssim_epoch = epoch

            logs = {
                "psnr": psnr_score,
                "ssim": ssim_score,
                "max_psnr": max_psnr,
                "max_ssim": max_ssim,
                "max_psnr_epoch": max_psnr_epoch,
                "max_ssim_epoch": max_ssim_epoch,
            }
            log_metrics(_run, logs, epoch, end_str="\n")

        # if (epoch + 1) % (opt.nEpochs * 2 / 3) == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 10.0
        #     print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

        #ll对学习率进行修改，第一种方式
        # if (epoch + 1) % (opt.nEpochs * 1 / 4) == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 2.0
        # elif (epoch + 1) % (opt.nEpochs * 1 / 2) == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 2.0
        # elif (epoch + 1) % (opt.nEpochs * 3 / 4) == 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] /= 2.0


        # ll对学习率进行修改，第二种方式，学习率退火衰减
        # 循环训练
        # lr_ll = 0.00015
        # #lr_ll = optimizer.param_groups[0]
        # # 余弦退火调整学习率
        # lr_ll *= 0.5 * (1.0 + math.cos(math.pi * epoch / opt.nEpochs))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = lr_ll
        # print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))


        # ll对学习率进行修改，第三种方式，余弦加预热
        # 设置学习率预热的参数
        lr_ll=0.0003
        lr_warmup = 0.00001
        warmup_epochs = 50  # 预热的epoch数
        if epoch < warmup_epochs:
            # 学习率预热
            lr_ll = lr_warmup + (lr_ll - lr_warmup) * epoch / warmup_epochs
        else:
            # 余弦退火调整学习率
            lr_ll *= 0.5 * (1.0 + math.cos(math.pi * (epoch - warmup_epochs) / (opt.nEpochs - warmup_epochs)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_ll
        print('G: Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))


