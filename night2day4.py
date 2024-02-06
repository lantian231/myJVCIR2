import os
import torch
from torchvision import transforms
from PIL import Image


class ImageBrightnessSampler:
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.probability_distribution = None

    def compute_brightness(self, folder_path):
        if self.probability_distribution is None:
            self.compute_probability_distribution(folder_path)

        sampled_brightness = self.sample_from_distribution(self.probability_distribution)

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

    def compute_probability_distribution(self, folder_path):
        image_files = self.get_image_files(folder_path)
        brightness_values = []

        for image_file in image_files:
            brightness = self.get_image_brightness(image_file)
            brightness_values.append(brightness)

        total_brightness = sum(brightness_values)
        self.probability_distribution = [brightness / total_brightness for brightness in brightness_values]

    def sample_from_distribution(self, probability_distribution):
        sampled_brightness = torch.distributions.Categorical(torch.tensor(probability_distribution)).sample().item()

        return sampled_brightness

# image_enhance = ImageBrightnessSampler()
# folder_path = 'datasets/LOL/train_data/high'
# d = image_enhance.compute_brightness(folder_path)
# d2 = d/(255.0*3.0)
# print(d)
# print(d2)