import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.nn import SyncBatchNorm

import numpy as np

from torchsummary import summary

from prism.utils.augmentations import (
    CustomColorJitter, Cutout, GaussianNoise,
    RandomErodeDilateTransform, ZeroMask, OnesMask
)


class Encoder(nn.Module):
    def __init__(self, input_channels, output_features):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adap_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, output_features)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.adap_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class CL(nn.Module):

    # note: h_dim means same thing as n_features
    def __init__(self, in_channels=5, h_dim=32, projection_dim=32, aug_params=None): 
        super(CL, self).__init__()
        self.aug = aug_params
        self.encoder = Encoder(input_channels=in_channels,output_features=h_dim)
        self.h_dim = h_dim
        self.base_size = 75

        #self.projection_dim = projection_dim
        #self.temperature = temperature #do we even need this?

        self.projector = nn.Sequential(
            nn.Linear(h_dim, h_dim, bias=False),
            nn.ReLU(),
            nn.Linear(h_dim, projection_dim, bias=False)
        ) 

    # x_i and x_j being the two augmented images of input x
    def forward(self, x):

        transform = self.simclr_transform()

        x_i = transform(x)
        x_j = transform(x)

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        return z_i, z_j, h_i, h_j
    
    def get_latent(self, x):
        return self.encoder(x)
    

    @staticmethod
    def loss(z_i, z_j, temperature):
        N = z_i.size(0)
        z = torch.cat((z_i, z_j), dim=0)
        z_normed = F.normalize(z, dim=1)
        cosine_similarity_matrix = torch.matmul(z_normed, z_normed.T)
        
        labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(z.device)
        
        # Remove self-similarity
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        cosine_similarity_matrix = cosine_similarity_matrix[~mask].view(cosine_similarity_matrix.shape[0], -1)

        positives = cosine_similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = cosine_similarity_matrix[~labels.bool()].view(labels.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)

        logits = logits / temperature
        loss = F.cross_entropy(logits, labels)

        return loss
        
    def simclr_transform(self):
        """Constructs the SimCLR data transformation pipeline."""
        transformations = []
        color_jitter = CustomColorJitter(brightness=self.aug["brightness"],
                                          contrast=self.aug["contrast"],
                                            saturation=self.aug["saturation"],
                                              hue=self.aug["hue"])  # Assuming brightness control as an example
        transformations.append(transforms.RandomApply([color_jitter],
                                                       p=1))
        transformations.append(transforms.RandomRotation(degrees=self.aug["rotation"]))
        transformations.append(transforms.RandomHorizontalFlip(p=self.aug["hflip"]))
        transformations.append(transforms.RandomVerticalFlip(p=self.aug["vflip"]))
        transformations.append(transforms.RandomAffine(degrees=0,
                                                        translate=(0.2,0.2)))
        blur = transforms.GaussianBlur(kernel_size=3,
                                        sigma=(0.1, 2.0))
        transformations.append(transforms.RandomApply([blur],
                                                      p=0.5))
        transformations.append(transforms.RandomResizedCrop(size=self.base_size,
                                                             scale=(0.8, 1.0)))
        data_transforms = transforms.Compose(transformations)
        return data_transforms