import torch
import argparse
import os, sys
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from DensePSA import Encoder, Decoder
from noiser import Noiser

class DataSet(Dataset):
    def __init__(self, image_folder):
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('.jpg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.CenterCrop(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img


def test():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset = 'mirflickr/test'
    args.device = 'cuda'
    args.batch_size = 16
    args.msg_size = 64
    
    testloader = torch.utils.data.DataLoader(DataSet(args.dataset), batch_size=args.batch_size, shuffle=True, drop_last=True)

    encoder = Encoder().to(args.device)
    decoder = Decoder().to(args.device)
    encoder.load_state_dict(torch.load('pretrain/DensePSA-encoder.pth', map_location=torch.device(args.device)))
    decoder.load_state_dict(torch.load('pretrain/DensePSA-decoder.pth', map_location=torch.device(args.device)))
    noiser = Noiser()

    print("start testing.")
    accu = []
    psnr = []
    for data in testloader:
        cover = data.to(args.device)
        origin_msg = torch.randint(0, 2, (args.batch_size,args.msg_size)).float().to(args.device)

        stego = encoder(cover, origin_msg).clamp(-1,1)
        noised = noiser.noise_list[8](stego)
        extract_msg = decoder(noised)

        accu.append(((extract_msg >= 0.5).eq(origin_msg >= 0.5).sum().float() / origin_msg.numel()).item())
        psnr.append(10 * torch.log10(4 / torch.mean((cover-stego)**2)).item())
    
    print('accu:', sum(accu)/len(accu), 'psnr:', sum(psnr)/len(psnr))


if __name__ == '__main__':
    test()
