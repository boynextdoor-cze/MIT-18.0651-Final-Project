import torch
from tqdm.notebook import tqdm
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import argparse
import wandb
import os
import copy
from attack import fgsm
from data.DataLoader import load_data
from utils import Avg_Metric, Normalize
from Denoising_autoencoder.model import Autoencoder
from randomize import random_reshape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(args, denoiser, model):
    model.eval()
    denoiser.eval()
    test_loader = load_data('test', batch_size=32, shuffle=False)
    test_accuracy = Avg_Metric()
    adv_model = copy.deepcopy(model)
    model.load_state_dict(torch.load('./checkpoints/fgsm_101_new.pth'))
    with torch.no_grad():
        for i, (data, labels, attack) in enumerate(tqdm(test_loader)):
            labels = labels.cuda()
            data = data.cuda()
            # Exclude wrongly classified examples
            logits = model(data)
            pred = torch.argmax(logits, dim=1)
            idx = (pred == labels).nonzero(as_tuple=True)[0]
            labels = labels[idx]
            data = data[idx]
            # attack = attack[idx].cuda()
            with torch.enable_grad():
                attack = fgsm(data, labels, adv_model,
                              nn.CrossEntropyLoss(), epsilon=0.2)
            if len(labels) == 0:
                continue
            if args.mode == 'random':
                anti_attack = random_reshape(attack)
            elif args.mode == 'denoise':
                anti_attack = denoiser(attack)
                mean, std = anti_attack.mean([2, 3]), anti_attack.std([2, 3])
                anti_attack = (anti_attack - mean[:, :, None, None]) / std[:, :, None, None]
            elif args.mode == 'None':
                mean, std = attack.mean([2, 3]), attack.std([2, 3])
                anti_attack = (attack - mean[:, :, None, None]) / std[:, :, None, None]
            else:
                raise ValueError('mode error')
            logits = model(anti_attack)
            # calculate accuracy
            pred = torch.argmax(logits, dim=1)
            acc = (pred == labels).sum().item()
            test_accuracy.update(acc, len(labels))
    print('Mode {} has test accuracy: {}'.format(
        args.mode, test_accuracy.avg * 100))


def train_val(args, denoiser, model):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.7)
    adv_model = copy.deepcopy(model)
    adv_model = adv_model.cuda()

    train_loader = load_data('train', batch_size=args.batch_size, shuffle=True)
    val_loader = load_data('val', batch_size=args.batch_size, shuffle=False)

    denoiser.eval()
    for epoch in range(args.train_epoch):
        model.train()
        train_acc = Avg_Metric()
        train_loss = Avg_Metric()
        adv_acc = Avg_Metric()
        for i, (data, labels, attack) in enumerate(train_loader):
            data = data.cuda()
            labels = labels.cuda()
            optimizer.zero_grad()
            mean, std = data.mean([2, 3]), data.std([2, 3])
            normal_data = (data - mean[:, :, None, None]) / std[:, :, None, None]
            logits = model(normal_data)
            # calculate accuracy
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == labels).item()
            train_acc.update(acc, len(labels))
            # calculate loss
            loss = criterion(logits, labels)
            train_loss.update(loss.item(), len(labels))
            loss.backward()
            optimizer.step()

            # Adversarial training
            optimizer.zero_grad()
            attack = fgsm(data, labels, adv_model, criterion, epsilon=0.2)
            mean, std = attack.mean([2, 3]), attack.std([2, 3])
            attack = (attack - mean[:, :, None, None]) / std[:, :, None, None]
            logits = model(attack)
            # calculate accuracy
            pred = torch.argmax(logits, dim=1)
            acc = torch.sum(pred == labels).item()
            adv_acc.update(acc, len(labels))
            # calculate loss
            loss = criterion(logits, labels)
            train_loss.update(loss.item(), len(labels))
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': train_loss.avg,
                      'train_acc': train_acc.avg * 100, 'adv_acc': adv_acc.avg * 100, 'step': (epoch + 1) * (i + 1)})
        scheduler.step()

        val_acc = Avg_Metric()
        val_loss = Avg_Metric()
        model.eval()
        with torch.no_grad():
            for i, (data, labels, attack) in enumerate(val_loader):
                data = data.cuda()
                labels = labels.cuda()
                with torch.enable_grad():
                    attack = fgsm(data, labels, adv_model,
                                  criterion, epsilon=0.2)
                mean, std = attack.mean([2, 3]), attack.std([2, 3])
                attack = (attack - mean[:, :, None, None]) / std[:, :, None, None]
                logits = model(attack)
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
            torch.save(model.state_dict(),
                       './checkpoints/epoch{}_101.pth'.format(epoch))


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

    model_bank = {'ResNet34': models.resnet34(weights=models.ResNet34_Weights.DEFAULT), 'ResNet101': models.resnet101(
        weights=models.ResNet101_Weights.DEFAULT), 'ResNet50': models.resnet50(weights=models.ResNet50_Weights.DEFAULT)}
    model = model_bank['ResNet101'].to(device)

    denoiser = Autoencoder().to(device)
    denoiser.load_state_dict(torch.load(
        './Denoising_autoencoder/epoch35_denoise_34.pth'))

    test(args, denoiser, model)


if __name__ == '__main__':
    main()
