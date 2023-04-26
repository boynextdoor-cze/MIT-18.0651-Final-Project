import torch
import torch.nn as nn
F = nn.functional
from torchvision import transforms

def random_reshape(imgs):
    """
    Randomly resize images from 224 to [310, 330] and pad to 331x331
    params:
        imgs: tensor of shape (N, C, H, W)
    """
    result = torch.zeros((imgs.shape[0], 3, 331, 331)).cuda()
    for i in range(imgs.shape[0]):
        img = imgs[i, :]
        resize = torch.randint(310, 331, (1,)).item()
        img = transforms.Resize(resize)(img)

        lr_pad = torch.randint(0, 331 - resize, (1,)).item()
        tb_pad = torch.randint(0, 331 - resize, (1,)).item()
        pad_size = (lr_pad, 331 - resize - lr_pad, tb_pad, 331 - resize - tb_pad)
        img = F.pad(img, pad_size, mode='constant', value=0)
        result[i, :] = img
    return result