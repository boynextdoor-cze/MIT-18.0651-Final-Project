import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from attack import fgsm

PROJ_DIR = os.getcwd()
TRAIN_VAL_PATH = os.path.join(PROJ_DIR, 'data', 'train_val')
TEST_PATH = os.path.join(PROJ_DIR, 'data', 'DAmageNet')
TRAIN_IMG_DIR = os.path.join(TRAIN_VAL_PATH, 'train')
VAL_IMG_DIR = os.path.join(TRAIN_VAL_PATH, 'val')
TEST_IMG_DIR = os.path.join(TEST_PATH, 'test')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def get_data_transform(split):
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])


class ImageNetDataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        sample.requires_grad = True

        target = torch.randint(1000, size=label.size())
        while target == label:
            target = torch.randint(1000, size=label.size())
        attack = fgsm(sample, target, epsilon=0.1)

        return sample, attack, label, target


def load_data(split, batch_size=64, num_workers=4, shuffle=True):
    txt = './data/{}/{}.txt'.format('train_val' if split ==
                                    'train' or split == 'val' else 'DAmageNet', split)

    print('Loading data from %s' % (txt))

    transform = get_data_transform(split)

    print('Use data transformation:', transform)

    if split == 'train':
        data_root = TRAIN_IMG_DIR
    elif split == 'val':
        data_root = VAL_IMG_DIR
    else:
        data_root = TEST_IMG_DIR

    dataset = ImageNetDataset(data_root, txt, transform)
    print('Length of {} set is: {}'.format(split, len(dataset)))

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
