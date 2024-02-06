import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
#from niqe import niqe

def calculate_metrics(ground_truth_folder, high_quality_folder):
    psnr_values = []
    ssim_values = []
    niqe_values = []

    for image_name in os.listdir(high_quality_folder):
        image_path = os.path.join(high_quality_folder, image_name)
        high_quality_image = cv2.imread(image_path)

        ground_truth_image_path = os.path.join(ground_truth_folder, image_name)
        ground_truth_image = cv2.imread(ground_truth_image_path)

        # Check if the images are valid
        if high_quality_image is None or ground_truth_image is None:
            print("Invalid image: {}".format(image_name))
            continue

        # Check if the images have the same size
        if high_quality_image.shape != ground_truth_image.shape:
            print("Images have different shapes: {}".format(image_name))
            continue

        # Calculate PSNR
        psnr_value = psnr(ground_truth_image, high_quality_image)
        psnr_values.append(psnr_value)
        print(f"psnr:{psnr_value}" )

        # Calculate SSIM
        ssim_value = ssim(ground_truth_image, high_quality_image, multichannel=True)
        ssim_values.append(ssim_value)
        print(f"ssim:{ssim_value}")
        # Calculate NIQE
        # niqe_value = niqe(high_quality_image)
        # niqe_values.append(niqe_value)

    # Calculate the average values
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    niqe_mean = np.mean(niqe_values)

    return psnr_mean, ssim_mean, niqe_mean


if __name__ == '__main__':
    gt_folder_path = 'datasets/LOL/eval15/high'
    hq_folder_path = 'datasets/LOL/eval15/result'
    #hq_folder_path = 'LLexperiments/last_result'
    #hq_folder_path = 'LLexperiments/every_result/result9'
    psnr, ssim, niqe = calculate_metrics(gt_folder_path, hq_folder_path)


    print("Average PSNR: {:.4f}".format(psnr))
    print("Average SSIM: {:.4f}".format(ssim))
    print("Average NIQE: {:.4f}".format(niqe))
