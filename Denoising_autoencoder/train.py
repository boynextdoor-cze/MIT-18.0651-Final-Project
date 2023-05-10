import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import tqdm
import wandb
import os
import torchvision.models as models

from model import Autoencoder

sys.path.append("..")
from data.DataLoader import load_data
from utils import Avg_Metric, ssim, Normalize
from attack import fgsm

if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = open(
        os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    wandb.init(project="matrix", name='autoencoder',
               config=dict(learing_rate=0.01, batch_size=32, epoch=50))

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")

    train_loader = load_data('train', apply_trans=True,
                             batch_size=32, num_workers=4, shuffle=True)
    val_loader = load_data('val', apply_trans=True,
                           batch_size=32, num_workers=4, shuffle=False)

    net = Autoencoder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)
    criterion = nn.MSELoss()
    alpha = 0.5
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT).to(device)

    for i in range(50):
        # training
        print(f'epoch-{i}-training begins')
        net.train()
        train_loss = Avg_Metric()
        for batch_idx, (sample, label, attack) in enumerate(train_loader):
            sample = sample.cuda()
            # attack = attack.cuda()
            attack = fgsm(sample, label, model, nn.CrossEntropyLoss(), epsilon=0.2)

            optimizer.zero_grad()
            output = net(attack)
            mse_loss = criterion(output, sample)
            ssim_loss = torch.mean(ssim(output, sample))
            loss = alpha * mse_loss + (1 - alpha) * ssim_loss
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), len(sample))
            wandb.log({"training_loss": train_loss.avg, 'step': (i * len(train_loader) + batch_idx)})
        scheduler.step()

        # validation
        print(f'epoch-{i}-validation begins')
        net.eval()
        val_loss = Avg_Metric()
        val_acc = Avg_Metric()
        with torch.no_grad():
            for batch_idx, (sample, labels, attack) in enumerate(val_loader):
                sample = sample.cuda()
                # attack = attack.cuda()
                with torch.enable_grad():
                    attack = fgsm(sample, labels, model, nn.CrossEntropyLoss(), epsilon=0.2)

                output = net(attack)
                mse_loss = criterion(output, sample)
                ssim_loss = torch.mean(ssim(output, sample))
                loss = alpha * mse_loss + (1 - alpha) * ssim_loss
                val_loss.update(loss.item(), len(sample))

                labels = labels.cuda()
                logits = model(sample)
                pred = torch.argmax(logits, dim=1)
                idx = (pred == labels).nonzero(as_tuple=True)[0]
                attack = attack[idx].cuda()
                labels = labels[idx]
                sample = sample[idx]
                if len(labels) == 0:
                    continue
                output = net(attack)
                mean, std = output.mean([2, 3]), output.std([2, 3])
                output = (output - mean[:, :, None, None]) / std[:, :, None, None]
                logits = model(output)
                pred = torch.argmax(logits, dim=1)
                acc = (pred == labels).sum().item()
                val_acc.update(acc, len(labels))
            wandb.log({"validation_loss": val_loss.avg, 'step': i})
            wandb.log({"validation_acc": val_acc.avg * 100, 'step': i})

        torch.save(net.state_dict(), "../checkpoints/epoch{}_denoise_101.pth".format(i))
