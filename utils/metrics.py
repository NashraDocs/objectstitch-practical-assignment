import torch
from lpips import LPIPS
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
import numpy as np
from PIL import Image

lpips_model = LPIPS(net='alex')

def pil_to_tensorrs(img):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(img).unsqueeze(0)

def Compute_lpipss(images1, images2):
    scores = []
    for img1, img2 in zip(images1, images2):
        t1 = pil_to_tensorrs(img1)
        t2 = pil_to_tensorrs(img2)
        score = lpips_model(t1, t2)
        scores.append(score.item())
    return sum(scores) / len(scores)

def Compute_ssim(img1, img2):
    image1_np = np.array(img1.resize((256, 256)).convert("L"))
    image2_np = np.array(img2.resize((256, 256)).convert("L"))
    return ssim(image1_np, image2_np)

def Compute_fids(fake_images, real_images):
    # Placeholder: assume FID is precomputed or skip
    # You can use pytorch-fid or tensorflow-fid CLI tools in practice
    return 0.0