import torch
from tqdm.notebook import tqdm
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import argparse
import wandb
import os
from attack import fgsm
from data.DataLoader import load_data
from utils import Avg_Metric
from Denoising_autoencoder.model import Autoencoder
from randomize import random_reshape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='none')
    parser.add_argument('-e', '--train_epoch', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-j', '--job', type=str, default=None)
    args = parser.parse_args()

    # os.environ['WANDB_API_KEY'] = open(
    #     os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    # wandb.init(project="matrix", name=args.job,
    #            config=wandb.helper.parse_config(args, exclude=['job']))
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    model_bank = {'InceptionV3': models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT), 'ResNet101': models.resnet101(
        weights=models.ResNet101_Weights.DEFAULT), 'ResNet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT)}
    model = model_bank['ResNet50'].to(device)
    denoiser = Autoencoder().to(device)
    # denoiser.load_state_dict(torch.load(
    #     './Denoising_autoencoder/epoch_10.pth'))

    test_loader = load_data('test', batch_size=1, shuffle=False)
    model.eval()
    test_accuracy = Avg_Metric()
    for i, (data, labels, attack) in enumerate(tqdm(test_loader)):
        attack = attack.to(device)
        if args.mode == 'random':
            anti_attack = random_reshape(attack)
        elif args.mode == 'denoise':
            anti_attack = denoiser(attack)
        elif args.mode == 'none':
            anti_attack = attack
        else:
            raise ValueError('mode error')
        labels = labels.to(device)
        logits = model(anti_attack)
        # calculate accuracy
        pred = torch.argmax(logits, dim=1)
        acc = (pred == labels).sum().item()
        test_accuracy.update(acc, len(labels))
    print('test accuracy: {}'.format(test_accuracy.avg * 100))

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=10, gamma=0.5)

    # train_loader = load_data('train', batch_size=args.batch_size, shuffle=True)
    # val_loader = load_data('val', batch_size=args.batch_size, shuffle=False)

    # for epoch in range(args.train_epoch):
    #     accuracy = Avg_Metric()
    #     losses = Avg_Metric()
    #     for i, (data, labels, attack) in enumerate(train_loader):
    #         if args.mode == 'random':
    #             anti_attack = random_reshape(attack).to(device)
    #         elif args.mode == 'denoise':
    #             anti_attack = denoiser(attack)
    #         elif args.mode == 'None':
    #             anti_attack = attack.to(device)
    #         else:
    #             raise ValueError('mode error')
    #         labels = labels.to(device)
    #         optimizer.zero_grad()
    #         logits = model(anti_attack)
    #         # calculate accuracy
    #         pred = torch.argmax(logits, dim=1)
    #         acc = (pred == labels).sum().item()
    #         accuracy.update(acc, len(labels))
    #         # calculate loss
    #         loss = criterion(logits, labels)
    #         losses.update(loss.item(), len(labels))
    #         loss.backward()
    #         optimizer.step()
    #     scheduler.step()
    #     wandb.log({'train_loss': losses.avg, 'train_acc': accuracy.avg, 'epoch': epoch})

if __name__ == '__main__':
    main()
