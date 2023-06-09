import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import cv2

PROJ_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGENET_PATH = os.path.join(PROJ_DIR, 'ImageNet')
DAMAGE_PATH = os.path.join(PROJ_DIR, 'DAmageNet')
TRAIN_IMG_DIR = os.path.join(IMAGENET_PATH, 'train')
VAL_IMG_DIR = os.path.join(IMAGENET_PATH, 'val')
TEST_IMG_DIR = os.path.join(IMAGENET_PATH, 'train')
DAMAGE_IMG_DIR = os.path.join(DAMAGE_PATH, 'test')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


class ImageNetDataset(Dataset):
    def __init__(self, root, txt, transform=None):
        super(ImageNetDataset, self).__init__()
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.attack_path = DAMAGE_IMG_DIR
        with open(txt) as f:
            for line in f:
                attack_img = os.path.join(self.attack_path, line.split()[0][:-4] + 'png')
                # attack_img = os.path.join(
                #     root, line.split()[0][:-5] + '_attack.JPEG')

                # if img does not exist, skip it
                if not os.path.exists(attack_img):
                    continue
                self.img_path.append({'origin': os.path.join(
                    root, line.split()[0]), 'attack': attack_img})
                self.labels.append(int(line.split()[1]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        origin_path = self.img_path[index]['origin']
        attack_path = self.img_path[index]['attack']
        label = self.labels[index]

        if self.transform is not None:
            with open(origin_path, 'rb') as f:
                sample = Image.open(f).convert('RGB')
                sample = self.transform(sample)
            with open(attack_path, 'rb') as f:
                attack = Image.open(f).convert('RGB')
                attack = self.transform(attack)
        else:
            sample = cv2.imread(origin_path)
            sample = cv2.resize(sample, (224, 224))
            sample = sample[:, :, ::-1].copy()
            sample = transforms.functional.to_tensor(sample)

            attack = cv2.imread(attack_path)
            attack = cv2.resize(attack, (224, 224))
            attack = attack[:, :, ::-1].copy()
            attack = transforms.functional.to_tensor(attack)

        return sample, label, attack


def load_data(split, apply_trans=True, batch_size=32, num_workers=4, shuffle=True):
    txt = os.path.join(PROJ_DIR, 'ImageNet/{}.txt'.format(split))

    print('Loading data from %s' % (txt))

    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        # normalize,
    ])

    if split == 'train':
        data_root = TRAIN_IMG_DIR
    elif split == 'val':
        data_root = VAL_IMG_DIR
    elif split == 'test':
        data_root = TEST_IMG_DIR
    else:
        raise ValueError('Unknown split: {}'.format(split))

    dataset = ImageNetDataset(
        data_root, txt, transform if apply_trans else None)
    print('Length of {} set is: {}'.format(split, len(dataset)))

    return DataLoader(dataset=dataset, batch_size=batch_size,
                      shuffle=shuffle, num_workers=num_workers)
