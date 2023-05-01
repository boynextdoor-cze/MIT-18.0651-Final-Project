import torch
import torch.nn as nn
F = nn.functional
from torchvision import transforms

def random_reshape(imgs):
    """
    Randomly resize images from 224 to [310, 350] and pad to 351x351
    params:
        imgs: tensor of shape (N, C, H, W)
    """
    output_shape = 351
    result = torch.zeros((imgs.shape[0], 3, output_shape, output_shape)).cuda()
    for i in range(imgs.shape[0]):
        img = imgs[i, :]
        resize = torch.randint(310, output_shape - 1, (1,)).item()
        img = transforms.Resize(resize)(img)

        lr_pad = torch.randint(0, output_shape - resize, (1,)).item()
        tb_pad = torch.randint(0, output_shape - resize, (1,)).item()
        pad_size = (lr_pad, output_shape - resize - lr_pad, tb_pad, output_shape - resize - tb_pad)
        img = F.pad(img, pad_size, mode='constant', value=0)
        result[i, :] = img
    return result