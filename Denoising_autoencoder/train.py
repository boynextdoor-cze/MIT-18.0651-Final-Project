from data.DataLoader import load_data
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import tqdm
import wandb
import os

from model import Autoencoder

sys.path.append("..")


if __name__ == '__main__':
    os.environ['WANDB_API_KEY'] = open(os.path.expandvars('$NFS/.wandb'), 'r').read().strip()
    wandb.init(project="Autoencoder", name='autoencoder',
               config=dict(learing_rate=0.01,batch_size=32,epoch=50))
    
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark=True

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU!")
    else:
        device = torch.device("cpu")
        print("Using CPU!")
    
    train_loader = load_data('train',apply_trans=True, batch_size=32, num_workers=4, shuffle=True)
    val_loader = load_data('val',apply_trans=True, batch_size=32, num_workers=4, shuffle=True)

    net = Autoencoder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    total_train_step = 0
    total_val_step = 0
    val_losses=[]
    train_losses=[]
    i=1
    training_epoch=0

    for i in range(50):
        # training
        print(f'epoch-{i}-training begins')
        net.train()
        for batch_idx, (sample, label, attack) in enumerate(train_loader):
            sample = sample.to(device)
            attack = attack.to(device)

            optimizer.zero_grad()
            output = net(attack)
            loss = criterion(output, sample)
            loss.backward()
            optimizer.step()

            # train_losses.append(loss.detach().cpu())
            wandb.log({"training_loss": loss.item(), 'step': training_epoch})
            training_epoch += 1
        if i==8:
            scheduler_1.step()
        if i==10:
            scheduler_1.step()
        if i==13:
            scheduler_1.step()
        if i>=15:
            scheduler_1.step()
        # print("the learning rate of the %dth epoch:%f" % (i, optimizer.param_groups[0]['lr']))

        # validation
        print(f'epoch-{i}-validation begins')
        net.eval()
        val_loss=0
        with torch.no_grad():
            for batch_idx, (sample, label, attack) in enumerate(val_loader):
                sample = sample.to(device)
                attack = attack.to(device)

                output = net(attack)
                loss = criterion(output, sample)
                val_loss+=loss.detach()
                # val_losses.append(loss.detach().cpu())
            val_loss=val_loss/len(val_loader)
            wandb.log({"validation_loss": val_loss.item(), 'step': i})

        torch.save(net,"epoch_{}.pth".format(i))