from torchvision import transforms
import torch
import torch.nn as nn
F = nn.functional


def random_reshape(imgs):
    """
    Randomly resize images from 224 to [224, 250] and pad to 331x331
    params:
        imgs: tensor of shape (N, C, H, W)
    """
    output_shape = 331
    result = torch.zeros((imgs.shape[0], 3, output_shape, output_shape)).cuda()
    for i in range(imgs.shape[0]):
        img = imgs[i, :].clone()
        resize = torch.randint(299, output_shape - 1, (1,)).item()
        img = transforms.Resize(resize)(img)

        lr_pad = torch.randint(0, output_shape - resize, (1,)).item()
        tb_pad = torch.randint(0, output_shape - resize, (1,)).item()
        pad_size = (lr_pad, output_shape - resize - lr_pad,
                    tb_pad, output_shape - resize - tb_pad)
        img = F.pad(img, pad_size, mode='constant', value=0)
        mean, std = img.mean([1, 2]), img.std([1, 2])
        img = transforms.Normalize(mean=mean, std=std)(img)
        result[i, :] = img
    return result
