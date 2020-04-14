import torch
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms

import os
import random
import numpy as np
import PIL.Image as Img

class SEAN_Dataset(Dataset):

    def __init__(self, data_path, status="Train"):

        self.data_path = data_path
        self.status = status

        self.dataset = self.get_data(self.data_path)

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):

        img_path, label_path = self.dataset[index]
        image = Img.open(img_path).convert("RGB")
        label = Img.open(label_path)

        need_flip = random.random() > 0.5
        self.label_transforms, self.img_transforms = self.get_transform(self.status, need_flip=need_flip)

        label_tensor = self.label_transforms(label) * 255.
        map = torch.zeros((19, 256, 256))

        final_map = map.scatter_(0, label_tensor.long(), 1)
        img_tensor = self.img_transforms(image)


        return  final_map, img_tensor

    def get_data(self, path):

        total_path = []

        for lable_name in os.listdir(path):
            lable_path = os.path.join(path, lable_name)
            image_path = os.path.join(path.replace("labels", "images"), lable_name.split(".")[0] + ".jpg")
            total_path.append([image_path, lable_path])

        return total_path

    def get_transform(self, status, need_flip):

        label_transform = []
        img_transform = []
        if status == "Train":# 随机flip
            label_transform.extend([transforms.Resize((256, 256), interpolation=Img.NEAREST),
                                    transforms.Lambda(lambda img: self.flip(img, need_flip)),
                                    transforms.ToTensor()
                                    ])
            img_transform.extend([transforms.Resize((256, 256), interpolation=Img.BICUBIC),
                                  transforms.Lambda(lambda img: self.flip(img, need_flip)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))
                                  ])

        elif status in ["Test","Validate"]: # 不需要flip
            label_transform.extend([transforms.Resize((256, 256), interpolation=Img.NEAREST),
                                    transforms.ToTensor()
                                    ])
            img_transform.extend([transforms.Resize((256, 256), interpolation=Img.BICUBIC),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                       (0.5, 0.5, 0.5))
                                  ])

        return transforms.Compose(label_transform), transforms.Compose(img_transform)

    def flip(self, img, need_flip):

        if need_flip:
            return img.transpose(Img.FLIP_LEFT_RIGHT)
        else:
            return img

if __name__ == "__main__":


    path = "/home/liuchao/SEAN-master/datasets/CelebA-HQ/train/labels"

    dataset = SEAN_Dataset(data_path=path)

    dataloader = DataLoader(dataset,batch_size=1,shuffle=True)

    for i, (seg, img) in enumerate(dataloader):
        print(seg.size())
        print(torch.unique(seg))
        print(img.size())
        print(torch.max(img), torch.min(img))



