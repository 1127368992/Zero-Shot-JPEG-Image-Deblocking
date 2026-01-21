import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os


def load_custom_256x256_image(img_path, is_gray=True):

    try:
        img = Image.open(img_path)
        if img.size != (256, 256):
            raise ValueError(f"{img.size[0]}×{img.size[1]}")
        if is_gray:
            img = img.convert('L')
        else:
            img = img.convert('RGB')

        img_array = np.array(img, dtype=np.uint8)
        return img_array

    except FileNotFoundError:
        raise FileNotFoundError(f"{img_path}")
    except Exception as e:
        raise Exception(f"{str(e)}")


def generate_jpeg_blocky_image(original_img, qf):

    pil_img = Image.fromarray(original_img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=qf)
    buffer.seek(0)
    jpeg_img = Image.open(buffer)
    blocky_img = np.array(jpeg_img)
    return blocky_img


def visualize_blocky_images(original_img, qf_list, blocky_imgs):
    plt.figure(figsize=(15, 8))
    plt.subplot(2, len(qf_list) // 2 + 1, 1)
    cmap = 'gray' if len(original_img.shape) == 2 else None
    plt.imshow(original_img, cmap=cmap, vmin=0, vmax=255)
    plt.title('Original Image (256×256)')
    plt.axis('off')
    for i, (qf, img) in enumerate(zip(qf_list, blocky_imgs)):
        plt.subplot(2, len(qf_list) // 2 + 1, i + 2)
        plt.imshow(img, cmap=cmap, vmin=0, vmax=255)
        plt.title(f'QF = {qf}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def save_blocky_images(blocky_imgs, qf_list, save_dir='./jpeg_blocky_images/'):

    os.makedirs(save_dir, exist_ok=True)
    for qf, img in zip(qf_list, blocky_imgs):
        save_path = os.path.join(save_dir, f'blocky_img_qf_{qf}.png')
        Image.fromarray(img).save(save_path)
        print(f'{save_path}')


if __name__ == '__main__':
    CUSTOM_IMG_PATH = "D:/BaiduNetdiskDownload/erwei/dataset/Set11/Parrots.tif"#D:/BaiduNetdiskDownload/erwei/dataset/Set11/barbara.tif
    IS_GRAY = False
    QF_LIST = [5,10,15,20,25,30,35,40,45,50]

    original_img = load_custom_256x256_image(CUSTOM_IMG_PATH, is_gray=IS_GRAY)

    blocky_imgs = []
    for qf in QF_LIST:
        blocky_img = generate_jpeg_blocky_image(original_img, qf)
        blocky_imgs.append(blocky_img)

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    for qf, img in zip(QF_LIST, blocky_imgs):
        psnr = peak_signal_noise_ratio(original_img, img, data_range=255)
        if IS_GRAY:
            ssim = structural_similarity(original_img, img, data_range=255)
        else:
            ssim = structural_similarity(original_img, img, data_range=255, channel_axis=-1)
        print(f"QF={qf} | PSNR={psnr:.2f}dB | SSIM={ssim:.4f}")

    visualize_blocky_images(original_img, QF_LIST, blocky_imgs)

    save_blocky_images(blocky_imgs, QF_LIST)