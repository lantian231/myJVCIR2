from __future__ import print_function
import argparse
import torch
from model_simplify229 import model
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
import math
from lib.dataset import is_image_file
from PIL import Image
from os import listdir

import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=1, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=2, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=256, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=16, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='datasets/LOL/eval15/low')
parser.add_argument('--model_type', type=str, default='SGN_DLN_new')
parser.add_argument('--output', default='datasets/LOL/eval15/result/', help='Location to save checkpoint models')
parser.add_argument('--modelfile', default='models_LOL_simplify229_smooth3\SGN_DLN_new_2070.pth', help='sr pretrained base model')
parser.add_argument('--image_based', type=bool, default=True, help='use image or video based ULN')
parser.add_argument('--chop', type=bool, default=False)

parser.add_argument('--output2', default='LLexperiments\LL_result', help='Location to save predict images for test_big')

opt = parser.parse_args()

gpus_list = range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Building model ', opt.model_type)

model = model()
#model = torch.nn.DataParallel(model, device_ids=gpus_list)
model.load_state_dict(torch.load(
    opt.modelfile,
    map_location=lambda storage, loc: storage))
if cuda:
    model = model.cuda(gpus_list[0])


def eval():
    model.eval()
    LL_filename = os.path.join(opt.image_dataset)
    est_filename = os.path.join(opt.output)
    big_result = os.path.join(opt.output2)
    try:
        os.stat(est_filename)
    except:
        os.mkdir(est_filename)

    try:
        os.stat(big_result)
    except:
        os.mkdir(big_result)

    LL_image = [join(LL_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    #print(LL_filename)
    Est_img = [join(est_filename, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    #print(Est_img)
    Big_img = [join(big_result, x) for x in sorted(listdir(LL_filename)) if is_image_file(x)]
    trans = transforms.ToTensor()
    channel_swap = (1, 2, 0)
    time_ave = 0
    for i in range(LL_image.__len__()):
        with torch.no_grad():
            LL_in = Image.open(LL_image[i]).convert('RGB')
            LL = trans(LL_in)
            LL = LL.unsqueeze(0)
            LL = LL.cuda()
            t0 = time.time()
            #_, _, prediction,_,= model(LL)
            _, add2, prediction,add7,add12,en1,en2= model(LL)
            #打印保存中间结果
            add2 = add2.data[0].cpu().numpy().transpose(channel_swap)
            add7 = add7.data[0].cpu().numpy().transpose(channel_swap)
            add12 = add12.data[0].cpu().numpy().transpose(channel_swap)
            en1 = en1.data[0].cpu().numpy().transpose(channel_swap)
            en2 = en2.data[0].cpu().numpy().transpose(channel_swap)

            add2 = add2 * 255.0
            add2 = add2.clip(0, 255)
            add7 = add7 * 255.0
            add7 = add7.clip(0, 255)
            add12 = add12 * 255.0
            add12 = add12.clip(0, 255)
            en1 = en1 * 255.0
            en1 = en1.clip(0, 255)
            en2 = en2 * 255.0
            en2 = en2.clip(0, 255)


            Image.fromarray(np.uint8(add2)).save(f"mid_result\\{i}_add2.png")
            Image.fromarray(np.uint8(add7)).save(f"mid_result\\{i}_add7.png")
            Image.fromarray(np.uint8(add12)).save(f"mid_result\\{i}_add12.png")
            Image.fromarray(np.uint8(en1)).save(f"mid_result\\{i}_en1.png")
            Image.fromarray(np.uint8(en2)).save(f"mid_result\\{i}_en2.png")

            #打印保存中间结果结尾
            t1 = time.time()
            time_ave += (t1 - t0)
            prediction = prediction.data[0].cpu().numpy().transpose(channel_swap)

            prediction = prediction * 255.0
            prediction = prediction.clip(0, 255)
            Image.fromarray(np.uint8(prediction)).save(Est_img[i])
            Image.fromarray(np.uint8(prediction)).save(Big_img[i])
            Image.fromarray(np.uint8(prediction)).save(f"mid_result\\{i}_pre.png")

            print("===> Processing Image: %04d /%04d in %.4f s." % (i, LL_image.__len__(), (t1 - t0)))

    print("===> Processing Time: %.4f ms." % (time_ave / LL_image.__len__() * 1000))


def modcrop(img, modulo):
    (ih, iw) = img.size
    ih = ih - (ih % modulo)
    iw = iw - (iw % modulo)
    img = img.crop((0, 0, ih, iw))

    return img


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        float32, [0, 255]
        float32, [0, 255]
    '''
    img.astype(np.float32)
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    rlt = rlt.round()

    return rlt


def PSNR(pred, gt, shave_border):
    pred = pred[shave_border:-shave_border, shave_border:-shave_border]
    gt = gt[shave_border:-shave_border, shave_border:-shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


transform = transforms.Compose([
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
]
)

##Eval Start!!!!
eval()
