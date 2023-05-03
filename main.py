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

cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")


def test(args, denoiser, model):
    model.eval()
    denoiser.eval()
    test_loader = load_data('test', batch_size=32, shuffle=False)
    test_accuracy = Avg_Metric()
    with torch.no_grad():
        for i, (data, labels, attack) in enumerate(tqdm(test_loader)):
            attack = attack.to(device)
            if args.mode == 'random':
                anti_attack = random_reshape(attack)
            elif args.mode == 'denoise':
                anti_attack = denoiser(attack)
            elif args.mode == 'None':
                anti_attack = attack
            else:
                raise ValueError('mode error')
            labels = labels.to(device)
            logits = model(anti_attack)
            # calculate accuracy
            pred = torch.argmax(logits, dim=1)
            acc = (pred == labels).sum().item()
            test_accuracy.update(acc, len(labels))
    print('Mode {} has test accuracy: {}'.format(
        args.mode, test_accuracy.avg * 100))


def train_val(args, denoiser, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.5)

    train_loader = load_data('train', batch_size=args.batch_size, shuffle=True)
    val_loader = load_data('val', batch_size=args.batch_size, shuffle=False)

    denoiser.eval()
    best_acc = 0
    for epoch in range(args.train_epoch):
        model.train()
        train_acc = Avg_Metric()
        train_loss = Avg_Metric()
        for i, (data, labels, attack) in enumerate(train_loader):
            attack = attack.to(device)
            if args.mode == 'random':
                anti_attack = random_reshape(attack)
            elif args.mode == 'denoise':
                anti_attack = denoiser(attack)
            elif args.mode == 'None':
                anti_attack = attack
            else:
                raise ValueError('mode error')
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(anti_attack)
            # calculate accuracy
            probs = F.softmax(logits, dim=1)
            _, pred = torch.max(probs, dim=1)
            acc = torch.sum(pred == labels).item()
            train_acc.update(acc, len(labels))
            # calculate loss
            loss = criterion(logits, labels)
            train_loss.update(loss.item(), len(labels))
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': train_loss.avg,
                      'train_acc': train_acc.avg * 100, 'step': (epoch + 1) * (i + 1)})
        scheduler.step()

        val_acc = Avg_Metric()
        val_loss = Avg_Metric()
        model.eval()
        with torch.no_grad():
            for i, (data, labels, attack) in enumerate(val_loader):
                attack = attack.to(device)
                if args.mode == 'random':
                    anti_attack = random_reshape(attack)
                elif args.mode == 'denoise':
                    anti_attack = denoiser(attack)
                elif args.mode == 'None':
                    anti_attack = attack
                else:
                    raise ValueError('mode error')
                labels = labels.to(device)
                logits = model(anti_attack)
                # calculate accuracy
                probs = F.softmax(logits, dim=1)
                _, pred = torch.max(probs, dim=1)
                acc = torch.sum(pred == labels).item()
                val_acc.update(acc, len(labels))
                # calculate loss
                loss = criterion(logits, labels)
                val_loss.update(loss.item(), len(labels))
            wandb.log({'val_loss': val_loss.avg, 'val_acc': val_acc.avg *
                      100, 'step': (epoch + 1) * (i + 1)})
            if val_acc.avg > best_acc:
                best_acc = val_acc.avg
                torch.save(model.state_dict(),
                           './checkpoints/epoch{}.pth'.format(epoch))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, default='None')
    parser.add_argument('-e', '--train_epoch', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-j', '--job', type=str, default=None)
    args = parser.parse_args()

    os.environ['WANDB_API_KEY'] = open(
        os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    wandb.init(project="matrix", name=args.job,
               config=wandb.helper.parse_config(args, exclude=['job']))
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    model_bank = {'InceptionV3': models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT), 'ResNet101': models.resnet101(
        weights=models.ResNet101_Weights.DEFAULT), 'ResNet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT)}
    model = model_bank['ResNet101'].to(device)

    denoiser = Autoencoder().to(device)
    denoiser.load_state_dict(torch.load('./Denoising_autoencoder/epoch25.pth'))

    train_val(args, denoiser, model)


if __name__ == '__main__':
    main()
