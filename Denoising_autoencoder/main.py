import torch
from tqdm import tqdm_notebook as tqdm
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import sys

sys.path.append("..")
from DataLoader import load_data
from utils import Accuracy


def main():
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    torch.manual_seed(42)
    BATCH_SIZE = 64

    classifiers = [{'model': models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device), 'name': 'resnet18'},
                 {'model':  models.resnet50(
                     weights=models.ResNet50_Weights.DEFAULT).to(device), 'name': 'resnet50'},
                 {'model':  models.resnet101(
                    weights=models.ResNet101_Weights.DEFAULT).to(device), 'name': 'resnet101'}]

    model = torch.load('epoch_25.pth', map_location=lambda storage, loc: storage)

    test_loader = load_data('test', batch_size=BATCH_SIZE)
    for classifier in classifiers:
        classifier['model'].eval()
        model.eval()
        attack = Accuracy()
        for i, (sample,labels,attacked_data) in enumerate(tqdm(test_loader)):
            with torch.no_grad():
                attacked_data,labels= attacked_data.to(device),labels.to(device)
                model.cuda()
                denoised_data = model(attacked_data)
                model.cpu()
                logits = classifier['model'](denoised_data)
                probs = F.softmax(logits, dim=1)
                _, pred = torch.max(probs, dim=1)
                attack.update(torch.sum(pred == labels).item(), BATCH_SIZE)
        print('Classifier: {}, Attack accuracy: {:.2f}%'.format(
                classifier['name'], attack.avg * 100))
        
if __name__ == '__main__':
    main()
