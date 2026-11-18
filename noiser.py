import numpy as np
import torch
from PIL import Image
import io
from kornia.filters import MedianBlur as korniaMedianBlur
from torchvision import transforms

def gaussian_kernel_2d(kernel_size, sigma):
    coords = torch.arange(kernel_size).float() - kernel_size // 2
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d /= kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d

class GaussianNoise():
    def __init__(self, std):
        self.std = std
    def __call__(self, image):
        return image + self.std * torch.randn_like(image)

class GaussianBlur():
    def __init__(self, kernel_size, sigma):
        self.kernel = gaussian_kernel_2d(kernel_size, sigma)[None, None, :]#.to('cuda')
        self.padding = (kernel_size-1)//2
    def __call__(self, image):
        C = image.shape[1]
        return torch.nn.functional.conv2d(image, self.kernel.repeat(C,1,1,1), padding=self.padding, groups=C)

class MedianBlur(korniaMedianBlur):
    def __init__(self, kernel_size):
        super().__init__((kernel_size, kernel_size))

class SaltPepper():
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, image):
        device = image.device
        dtype = image.dtype
        rand = torch.rand(image.shape, device = device)
        salt_thresh = pepper_thresh = self.prob*0.5
        salt_mask = rand < salt_thresh
        pepper_mask = (rand >= salt_thresh) & (rand < (salt_thresh + pepper_thresh))
        noisy_image = torch.where(salt_mask, torch.tensor(1.0, device=device, dtype=dtype), image)
        noisy_image = torch.where(pepper_mask, torch.tensor(-1.0, device=device, dtype=dtype), noisy_image)
        return noisy_image



def jpeg_transformation(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    return Image.open(buffer)

class ForwardJPEG():
    def __init__(self):
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(jpeg_transformation),
            transforms.ToTensor()
        ])
    def __call__(self, image):
        processed_images = []
        image_cpu = (image.detach().cpu()+1)/2
        for i in range(image_cpu.shape[0]):
            single_img = self.trans(image_cpu[i])
            processed_images.append(single_img*2-1)
        return (torch.stack(processed_images, dim=0).to(image.device)-image).detach() + image

class Noiser():
    def __init__(self):
        self.noise_list = [
            GaussianBlur(7, 2),
            GaussianBlur(5, 2),
            GaussianBlur(3, 2),
            GaussianNoise(0.1),
            SaltPepper(0.05),
            MedianBlur(7),
            MedianBlur(5),
            MedianBlur(3),
            ForwardJPEG()
        ]
    def __call__(self, image):
        n = np.random.choice(self.noise_list, p=[1/15,1/15,1/15,1/5,1/5,1/15,1/15,1/15,1/5])
        return n(image.clamp(-1,1)).clamp(-1,1)
