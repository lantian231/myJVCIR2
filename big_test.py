import os
import shutil
import subprocess
import cv2

def calculate_psnr(image1_path, image2_path):
    # 读取图片
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # 计算MSE
    mse = ((image1 - image2) ** 2).mean()

    # 计算PSNR
    psnr = cv2.PSNR(image1, image2)

    return psnr


def main():
    #定义迭代次数
    iter = 10
    # 运行 test.py 五次，并将结果保存到对应的文件夹中
    for i in range(iter):
        subprocess.run(["python", "test.py"])

        # 将生成的结果图片移动到对应的结果文件夹

        source_folder = "LLexperiments/LL_result"
        destination_folder = "LLexperiments/every_result/"+f"result{i+1}"
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        for root, dirs, files in os.walk(source_folder):
            for file in files:
                source_path = os.path.join(root, file)
                destination_path = os.path.join(destination_folder, file)
                shutil.copy2(source_path, destination_path)
                print(f"Copied: {source_path} -> {destination_path}")
                #shutil.move(source_path, destination_path)

    #从每次的实验结果中挑选最优的图片，存到最终的结果中

    #擂台法，先将第一个子文件夹中的所有图片结果存到最终结果文件夹中
    source_folder = "LLexperiments/every_result/" + f"result1"
    destination_folder = "LLexperiments/last_result"
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for root, dirs, files in os.walk(source_folder):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy2(source_path, destination_path)
            print(f"init_search_best: {source_path} -> {destination_path}")
    for i in range(iter):
        # 获取挑战者和应战者文件夹中的所有文件名
        folder1_path = "LLexperiments/every_result/" + f"result{i+1}"
        folder2_path = "LLexperiments/last_result"
        folder1_files = os.listdir(folder1_path)
        #folder2_files = os.listdir(folder2_path)
        for image in folder1_files:#遍历挑战者和应战者文件夹中的所有图片，计算psnr
            psnr1 = calculate_psnr("LLexperiments/every_result/" + f"result{i+1}/"+f"{image}","datasets/LOL/eval15/high/"+f"{image}")    #挑战者的psnr
            print(f"psnr1={psnr1}")
            psnr2 = calculate_psnr("LLexperiments/last_result/" + f"{image}","datasets/LOL/eval15/high/"+f"{image}")                     #应战者的psnr
            print(f"psnr2={psnr2}")
            if psnr1>psnr2:             #如果挑战者更厉害，就让挑战者做擂主
                # 使用shutil库进行文件复制
                shutil.copy("LLexperiments/every_result/" + f"result{i+1}/"+f"{image}", "LLexperiments/last_result/" + f"{image}")
                print(f"Image copied to experiments/last_result")

    subprocess.run(["python", "eval_ll2.py"])

if __name__ == "__main__":
    main()
