import os
import time
import argparse
import numpy as np
from PIL import Image
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.init as init

import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image

import einops

parser = argparse.ArgumentParser('Zero-Shot Image Deblocking')
parser.add_argument('--data_path', default='./dataset', type=str, help='Path to the data')
parser.add_argument('--dataset', default='kuai', type=str, help='Dataset name')
parser.add_argument('--save', default='./block_banks', type=str, help='Directory to save block bank results')
parser.add_argument('--out_image', default='./deblocked_results', type=str, help='Directory to save deblocked images')
parser.add_argument('--ws', default=32, type=int, help='Window size for searching similar blocks')
parser.add_argument('--bs', default=8, type=int, help='Block size (typically 8 for JPEG)')
parser.add_argument('--nn', default=12, type=int, help='Number of nearest neighbors to search')
parser.add_argument('--mm', default=6, type=int, help='Number of blocks in block bank to use for training')
parser.add_argument('--qf', default=20, type=int, help='Quality factor for JPEG compression (lower = more blocking)')
parser.add_argument('--loss', default='L1', type=str, help='Loss function type')
args = parser.parse_args()

torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"

WINDOW_SIZE = args.ws
BLOCK_SIZE = args.bs
NUM_NEIGHBORS = args.nn
quality_factor = args.qf
loss_type = args.loss

transform = transforms.Compose([transforms.ToTensor()])


def add_blocking_artifact(x, qf=15):

    x_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
    x_pil = Image.fromarray(x_np.astype(np.uint8))

    import io as bytesio
    buffer = bytesio.BytesIO()
    x_pil.save(buffer, format='JPEG', quality=qf)
    buffer.seek(0)
    blocked_pil = Image.open(buffer)
    blocked = transform(blocked_pil).unsqueeze(0)
    return blocked

def construct_block_bank():
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.qf, args.ws, args.bs, args.nn, args.loss]))
    os.makedirs(bank_dir, exist_ok=True)

    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    pad_sz = WINDOW_SIZE // 2 + BLOCK_SIZE // 2
    center_offset = WINDOW_SIZE // 2
    blk_sz = 64  

    for image_file in image_files:
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        image_path = os.path.join(image_folder, image_file)
        start_time = time.time()

        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256), Image.Resampling.LANCZOS)
        img_clean = transform(img).unsqueeze(0)  # [1, C, H, W]
        img_blocked = add_blocking_artifact(img_clean, qf=quality_factor)
        img_blocked = img_blocked.to(device)

 
        img_pad = F.pad(img_blocked, (pad_sz, pad_sz, pad_sz, pad_sz), mode='reflect')
        img_unfold = F.unfold(img_pad, kernel_size=BLOCK_SIZE, padding=0, stride=1)
        H_unfold = img_pad.shape[-2] - BLOCK_SIZE + 1
        W_unfold = img_pad.shape[-1] - BLOCK_SIZE + 1
        img_unfold = einops.rearrange(img_unfold, 'b c (h w) -> b c h w', h=H_unfold, w=W_unfold)

        num_blk_w = img_blocked.shape[-1] // blk_sz  # 256/64=4
        num_blk_h = img_blocked.shape[-2] // blk_sz  # 256/64=4
        is_window_size_even = (WINDOW_SIZE % 2 == 0)
        topk_list = []

        for blk_i in range(num_blk_w):
            for blk_j in range(num_blk_h):
                start_h = blk_j * blk_sz
                end_h = (blk_j + 1) * blk_sz + WINDOW_SIZE
                start_w = blk_i * blk_sz
                end_w = (blk_i + 1) * blk_sz + WINDOW_SIZE

                end_h = min(end_h, img_unfold.shape[-2])
                end_w = min(end_w, img_unfold.shape[-1])
                sub_img_uf = img_unfold[..., start_h:end_h, start_w:end_w]
                sub_img_shape = sub_img_uf.shape

                if is_window_size_even:
                    sub_img_uf_inp = sub_img_uf[..., :-1, :-1]
                else:
                    sub_img_uf_inp = sub_img_uf

                try:
                    patch_windows = F.unfold(sub_img_uf_inp, kernel_size=WINDOW_SIZE, padding=0, stride=1)
                    patch_windows = einops.rearrange(
                        patch_windows,
                        'b (c k1 k2 k3 k4) (h w) -> b (c k1 k2) (k3 k4) h w',
                        k1=BLOCK_SIZE, k2=BLOCK_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE,
                        h=blk_sz, w=blk_sz
                    )
                except Exception as e:
                    print(f"{e}")
                    continue

                img_center = einops.rearrange(
                    sub_img_uf,
                    'b (c k1 k2) h w -> b (c k1 k2) 1 h w',
                    k1=BLOCK_SIZE, k2=BLOCK_SIZE,
                    h=sub_img_shape[-2], w=sub_img_shape[-1]
                )
                end_h_center = min(center_offset + blk_sz, img_center.shape[-2])
                end_w_center = min(center_offset + blk_sz, img_center.shape[-1])
                img_center = img_center[..., center_offset:end_h_center, center_offset:end_w_center]

                if args.loss == 'L2':
                    distance = torch.sum((img_center - patch_windows) ** 2, dim=1)
                elif args.loss == 'L1':
                    distance = torch.sum(torch.abs(img_center - patch_windows), dim=1)
                else:
                    raise ValueError(f"Unsupported loss type: {loss_type}")

                _, sort_indices = torch.topk(
                    distance,
                    k=min(NUM_NEIGHBORS, distance.shape[-3]), 
                    largest=False,
                    sorted=True,
                    dim=-3
                )

                patch_windows_reshape = einops.rearrange(
                    patch_windows,
                    'b (c k1 k2) (k3 k4) h w -> b c (k1 k2) (k3 k4) h w',
                    k1=BLOCK_SIZE, k2=BLOCK_SIZE, k3=WINDOW_SIZE, k4=WINDOW_SIZE
                )
                patch_center = patch_windows_reshape[:, :, patch_windows_reshape.shape[2] // 2, ...]
                topk = torch.gather(patch_center, dim=-3,
                                    index=sort_indices.unsqueeze(1).repeat(1, 3, 1, 1, 1))
                topk_list.append(topk)

        if not topk_list:
            print(f" {image_file}")
            continue

        topk = torch.cat(topk_list, dim=0)
        topk = einops.rearrange(topk, '(w1 w2) c k h w -> k c (w2 h) (w1 w)',
                                w1=num_blk_w, w2=num_blk_h)
        topk = topk.permute(2, 3, 0, 1)

        elapsed = time.time() - start_time
        print(f"Processed {image_file} in {elapsed:.2f} seconds. Block bank shape: {topk.shape}")

        file_name_without_ext = os.path.splitext(image_file)[0]
        np.save(os.path.join(bank_dir, file_name_without_ext), topk.cpu())

    print("Block bank construction completed for all images.")


class DeblockNetwork(nn.Module):
    def __init__(self, n_chan, chan_embed=64):
        super(DeblockNetwork, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv5 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv6 = nn.Conv2d(chan_embed, n_chan, 3, padding=1)
        self._initialize_weights()

    def forward(self, x):
        residual = x
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x_res = self.act(self.conv3(x))
        x_res = self.conv4(x_res)
        x = x + x_res
        x = self.act(self.conv5(x))
        x = self.conv6(x)
        return torch.clamp(x + residual, 0, 1)  

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


loss_f = nn.L1Loss() if args.loss == 'L1' else nn.MSELoss()


def loss_func(blocked_patch, clean_patch, model, loss_f):
    pred = model(blocked_patch)
    loss = loss_f(clean_patch, pred)
    return loss


def train(model, optimizer, block_bank):
    N, H, W, C = block_bank.shape
    index1 = torch.randint(0, N, size=(H, W), device=device)
    index1_exp = index1.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    block1 = torch.gather(block_bank, 0, index1_exp)  # [1, H, W, C]
    block1 = block1.permute(0, 3, 1, 2)

    index2 = torch.randint(0, N, size=(H, W), device=device)
    eq_mask = (index2 == index1)
    if eq_mask.any():
        index2[eq_mask] = (index2[eq_mask] + 1) % N
    index2_exp = index2.unsqueeze(0).unsqueeze(-1).expand(1, H, W, C)
    block2 = torch.gather(block_bank, 0, index2_exp)
    block2 = block2.permute(0, 3, 1, 2)

    loss = loss_func(block1, block2, model, loss_f)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, blocked_img, clean_img):
    with torch.no_grad():
        pred = model(blocked_img)
        pred = torch.clamp(pred, 0, 1)
        mse_val = F.mse_loss(clean_img, pred).item()
        psnr = 10 * np.log10(1 / mse_val)
    return psnr, pred


def deblock_images():
    bank_dir = os.path.join(args.save, '_'.join(
        str(i) for i in [args.dataset, args.qf, args.ws, args.bs, args.nn, args.loss]))
    image_folder = os.path.join(args.data_path, args.dataset)
    image_files = sorted(os.listdir(image_folder))

    os.makedirs(args.out_image, exist_ok=True)

    max_epoch = 5000
    lr = 0.0005
    avg_PSNR = 0
    avg_SSIM = 0
    count = 0

    for image_file in image_files:
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue

        count += 1
        image_path = os.path.join(image_folder, image_file)
        clean_img = Image.open(image_path).convert('RGB')
        clean_img = clean_img.resize((256, 256), Image.Resampling.LANCZOS)
        clean_img_tensor = transform(clean_img).unsqueeze(0).to(device)
        clean_img_np = np.array(clean_img)

        blocked_img = add_blocking_artifact(clean_img_tensor, qf=quality_factor).to(device)

        file_name_without_ext = os.path.splitext(image_file)[0]
        bank_path = os.path.join(bank_dir, file_name_without_ext)
        if not os.path.exists(bank_path + '.npy'):
            print(f"Block bank for {image_file} not found, skipping.")
            continue

        block_bank_arr = np.load(bank_path + '.npy')
        if block_bank_arr.ndim == 3:
            block_bank_arr = np.expand_dims(block_bank_arr, axis=1)
        block_bank = block_bank_arr.astype(np.float32).transpose((2, 0, 1, 3))
        if quality_factor < 30:
            args.mm = 8  
        else:
            args.mm = 4
        block_bank = block_bank[:args.mm]
        block_bank = torch.from_numpy(block_bank).to(device)

        n_chan = clean_img_tensor.shape[1]
        model = DeblockNetwork(n_chan).to(device)
        print(f"Number of parameters for {image_file}: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=[1000, 1500], gamma=0.5)

        for epoch in range(max_epoch):
            loss = train(model, optimizer, block_bank)
            scheduler.step()
            if (epoch + 1) % 500 == 0:
                print(f"Epoch {epoch + 1}/{max_epoch}, Loss: {loss:.6f}")

        PSNR, out_img = test(model, blocked_img, clean_img_tensor)
        out_img_pil = to_pil_image(out_img.squeeze(0).cpu())
        out_img_save_path = os.path.join(args.out_image, file_name_without_ext + '_deblocked.png')
        out_img_pil.save(out_img_save_path)

        blocked_img_pil = to_pil_image(blocked_img.squeeze(0).cpu())
        blocked_img_save_path = os.path.join(args.out_image, file_name_without_ext + '_blocked.png')
        blocked_img_pil.save(blocked_img_save_path)

        out_img_np = np.array(out_img_pil)
        SSIM, _ = compare_ssim(
            clean_img_np,
            out_img_np,
            full=True,
            channel_axis=2,
            win_size=5,
            data_range=255
        )
        print(f"Image: {image_file} | PSNR: {PSNR:.2f} dB | SSIM: {SSIM:.4f}")
        avg_PSNR += PSNR
        avg_SSIM += SSIM

    if count > 0:
        avg_PSNR /= count
        avg_SSIM /= count
        print(f"Average PSNR: {avg_PSNR:.2f} dB, Average SSIM: {avg_SSIM:.4f}")


if __name__ == "__main__":
    print("Constructing block banks ...")
    construct_block_bank()
    print("Starting deblocking ...")
    deblock_images()