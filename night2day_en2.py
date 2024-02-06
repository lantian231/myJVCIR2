import os
import torch
from torchvision.transforms import functional as F
from PIL import Image

class BrightnessCalculator:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.average_brightness = None

    def compute_brightness(self):
        if self.average_brightness is not None:
            return self.average_brightness

        total_brightness = 0.0
        num_images = 0

        # 遍历文件夹中的图像文件
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.folder_path, filename)

                # 加载图像并转换为张量
                image = F.to_tensor(F.resize(Image.open(image_path), size=(224, 224)))
                image = image.unsqueeze(0)  # 添加批次维度

                # 计算亮度
                gray_image = torch.mean(image, dim=(2, 3))
                brightness = torch.mean(gray_image).item()

                total_brightness += brightness
                num_images += 1

        # 计算平均亮度
        if num_images > 0:
            self.average_brightness = total_brightness / num_images
            return self.average_brightness
        else:
            return None

# 使用示例
folder_path = "datasets/LOL/high_with_low"
calculator = BrightnessCalculator(folder_path)
avg_brightness = calculator.compute_brightness()
if avg_brightness is not None:
    print("Average brightness:", avg_brightness)
else:
    print("No images found in the folder.")

# 第二次调用，直接返回已保存的结果
avg_brightness = calculator.compute_brightness()
if avg_brightness is not None:
    print("Average brightness (from saved result):", avg_brightness)
else:
    print("No images found in the folder.")
