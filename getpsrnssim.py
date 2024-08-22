import os
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_psnr_and_ssim(folder1, folder2):
    # 遍历第一个文件夹中的所有图片
    psnr_total = 0
    ssim_total = 0
    num=0
    for filename in os.listdir(folder1):
        num+=1
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # 只处理图片文件
            # 读取两个文件夹中的图片
            img1_path = os.path.join(folder1, filename)
            img2_path = os.path.join(folder2, filename)

            # 确保第二个文件夹中存在对应的图片
            if not os.path.exists(img2_path):
                print(f"Warning: No corresponding image found for {filename} in the second folder.")
                continue

            img1 = img_as_float(io.imread(img1_path))
            img2 = img_as_float(io.imread(img2_path))

            # 计算PSNR
            psnr_value = psnr(img1, img2, data_range=img1.max() - img1.min())

            # 计算SSIM
            ssim_value = ssim(img1, img2, multichannel=True)
            psnr_total += psnr_value
            ssim_total += ssim_value

            print(f"File: {filename}")
            print(f"PSNR: {psnr_value:.4f} dB")
            print(f"SSIM: {ssim_value:.4f}")
            print("-" * 40)
    average_psnr = psnr_total/num
    average_ssim = ssim_total/num
    print(f'平均PSNR={average_psnr}, 平均SSIM={average_ssim}')



# 调用函数，传入两个文件夹路径
calculate_psnr_and_ssim(r"E:\zyl\LGTD-main\LGTD-main\Results\Ours-final-epoch9-208\b_4x\0211", r"E:\zyl\LGTD-main\LGTD-main\Jilin-189\trainset\training_set\GT\003")