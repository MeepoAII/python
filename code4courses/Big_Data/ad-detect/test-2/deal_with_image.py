import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np


data_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='/media/meepo/c2cb98a3-f300-42f6-860c-4acd5dc194a4/me'
                                          'epo/data/bigdata/'
                                          'my-experiment/try-1/data/train',transform=data_transform)
train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=16)


def show_batch_images(sample_batch):
    labels_batch = sample_batch[1]
    images_batch = sample_batch[0]

    for i in range(4):
        label_ = labels_batch[i].item()
        image_ = np.transpose(images_batch[i], (1, 2, 0))
        ax = plt.subplot(1, 4, i + 1)
        ax.imshow(image_)
        ax.set_title(str(label_))
        ax.axis('off')
        plt.pause(0.01)


plt.figure()
for i_batch, sample_batch in enumerate(train_dataloader):
    show_batch_images(sample_batch)

    plt.show()