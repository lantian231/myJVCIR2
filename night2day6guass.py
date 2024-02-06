import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

class ImageBrightnessSampler:
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.brightness_values = []
        self.mean_brightness = None
        self.variance = None

    def compute_brightness(self, folder_path):
        if not self.brightness_values:
            self.compute_brightness_values(folder_path)
            print("Gauss create!")
        if self.variance is None:
            self.compute_variance_and_stddev()

        sampled_brightness = self.sample_from_gaussian_distribution()

        return sampled_brightness

    def get_image_files(self, folder_path):
        image_files = []
        extensions = ['.jpg', '.jpeg', '.png']

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in extensions):
                image_files.append(os.path.join(folder_path, file))

        return image_files

    def get_image_brightness(self, image_file):
        image = Image.open(image_file).convert('RGB')
        image_tensor = self.transform(image)
        gray_tensor = self.rgb_to_gray(image_tensor)
        brightness = torch.mean(gray_tensor)

        return brightness.item()

    def rgb_to_gray(self, rgb_tensor):
        r, g, b = rgb_tensor[0], rgb_tensor[1], rgb_tensor[2]
        gray_tensor = 0.2989 * r + 0.587 * g + 0.114 * b

        return gray_tensor

    def compute_brightness_values(self, folder_path):
        image_files = self.get_image_files(folder_path)
        self.brightness_values = []

        for image_file in image_files:
            brightness = self.get_image_brightness(image_file)
            self.brightness_values.append(brightness)

    def compute_variance_and_stddev(self):
        if not self.brightness_values:
            raise ValueError("No brightness values computed. Run compute_brightness_values() first.")

        brightness_values = np.array(self.brightness_values)
        self.mean_brightness = np.mean(brightness_values)
        self.variance = np.var(brightness_values)

    def sample_from_gaussian_distribution(self):
        if self.variance is None:
            raise ValueError("Variance not computed. Run compute_variance_and_stddev() first.")

        if self.variance == 0:
            return np.random.choice(self.brightness_values)

        sampled_brightness = np.random.normal(self.mean_brightness, np.sqrt(self.variance))
        # Clip sampled brightness to ensure it falls within the valid range [0, 255]
        sampled_brightness = max(0, min(255, sampled_brightness))

        return sampled_brightness
