import torch
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import cv2
from attack import fgsm
from defense import autoencoder, random_clip
from DataLoader import load_data


cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
torch.manual_seed(42)
BATCH_SIZE = 64

classifiers = [{'model': models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device), 'name': 'resnet18'},
               {'model':  models.resnet50(
                   weights=models.ResNet50_Weights.DEFAULT).to(device), 'name': 'resnet50'},
               {'model':  models.resnet101(
                   weights=models.ResNet101_Weights.DEFAULT).to(device), 'name': 'resnet101'}]


class Accuracy(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, num, n):
        self.sum += num
        self.count += n
        self.avg = self.sum / self.count


# FGSM attack
train_loader = load_data('train', batch_size=BATCH_SIZE)
for classifier in classifiers:
    classifier.eval()
    no_attack = Accuracy()
    attack = Accuracy()
    fool = Accuracy()
    for i, (data, attacked_data, labels, targets) in enumerate(tqdm(train_loader)):
        data, labels = data.to(device), labels.to(device)
        attacked_data, targets = attacked_data.to(device), targets.to(device)
        outputs = classifier['model'](data)
        logits = F.softmax(outputs, dim=1)
        _, pred = torch.max(logits, dim=1)
        no_attack.update(torch.sum(pred == labels).item(), BATCH_SIZE)

        outputs = classifier['model'](attacked_data)
        logits = F.softmax(outputs, dim=1)
        _, pred = torch.max(logits, dim=1)
        attack.update(torch.sum(pred == labels).item(), BATCH_SIZE)
        fool.update(torch.sum(pred == targets).item(), BATCH_SIZE)
    print('Classifier: {}, No attack accuracy: {:.2f}%, Attack accuracy: {:.2f}%, Fool accuracy: {:.2f}%'.format(
        classifier['name'], no_attack.avg * 100, attack.avg * 100, fool.avg * 100))
