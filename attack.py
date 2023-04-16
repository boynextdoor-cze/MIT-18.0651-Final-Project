import torch
from tqdm import tqdm_notebook as tqdm
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def fgsm(x, labels, model, loss, epsilon=0.1):
    """
    :param x: input tensor.
    :param class_id: the id of the target class.
    :return: a tensor for the adversarial example
    """
    x = x.to(device)
    labels = labels.to(device)
    x.requires_grad = True
    
    outputs = model(x)
    
    cost = loss(outputs, labels).to(device)
    model.zero_grad()
    cost.backward()
    
    attack_images = x + epsilon*x.grad.data.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images
