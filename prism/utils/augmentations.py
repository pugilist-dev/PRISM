import torch
from torchvision import transforms

import numpy as np
import cv2

import random

class CustomColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
    
    def __call__(self, imgs):
        # imgs should be a batch of images with the shape [B, C, H, W]
        # B: Batch size, C: Number of channels (4 or 5 in your case), H: Height, W: Width
        batch_jittered = []
        for img in imgs:  # Loop over each image in the batch
            channels = []
            num_channels = img.shape[0]
            for i in range(min(num_channels, 4)):  # Loop over the first 4 channels
                single_channel = img[i].unsqueeze(0)  # Reshape [H, W] -> [1, H, W]
                jittered_channel = self.color_jitter(single_channel)  # Apply color jitter
                channels.append(jittered_channel.squeeze(0))  # Remove the added dimension
            if num_channels == 5:  # Check if the mask channel exists
                mask_channel = img[4].unsqueeze(0)  # Extract the mask channel and keep its shape
                channels.append(mask_channel.squeeze(0))  # Add the mask channel back to the list
            jittered_img = torch.stack(channels, dim=0)  # Stack all channels back together to form an image [C, H, W]
            batch_jittered.append(jittered_img)  # Append the processed image to the batch
        return torch.stack(batch_jittered, dim=0)  # Stack all images back into a batch [B, C, H, W]

class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes # n_holes (int): Number of holes to cut out from each image.
        self.length = length #  length (int): Length of each square hole.

    def __call__(self, imgs):
        # imgs (Tensor): Batch of images with shape (B, C, H, W).
        # returns batch of images with holes cut out
        batch_size, _, h, w = imgs.size()
        for i in range(batch_size):
            img = imgs[i]
            mask = np.ones((h, w), np.float32)
            for _ in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                y2 = np.clip(y + self.length // 2, 0, h)
                x2 = np.clip(x + self.length // 2, 0, w)

                mask[y1: y2, x1: x2] = 0

            mask = torch.from_numpy(mask).to("cuda:1")
            mask = mask.expand_as(img)
            imgs[i] *= mask

        return imgs

    def __repr__(self):
        return f"{self.__class__.__name__}(n_holes={self.n_holes}, length={self.length})"

class GaussianNoise(object):
    def __init__(self, mean=0., std=1.):
       
        # mean (float): Mean of the Gaussian noise.
        # std (float): Standard deviation of the Gaussian noise.
        
        self.mean = mean
        self.std = std

    def __call__(self, tensor):

        # tensor (Tensor): Batch of images with shape (B, C, H, W).
        # Generate Gaussian noise, returns batch with noise added 
        noise = torch.randn_like(tensor).to("cuda:1") * self.std + self.mean
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class RandomErodeDilateTransform:
    def __init__(self, kernel_size=5, iterations=1):
        self.kernel_size = kernel_size
        self.iterations = iterations
    
    def __call__(self, tensor):
        if tensor.shape[1] != 5:
            raise ValueError("The input tensor must have 5 channels")
                
        batch_size = tensor.shape[0]
        
        # Create a circular kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size), np.uint8)
        center = (self.kernel_size // 2, self.kernel_size // 2)
        radius = self.kernel_size // 2
        cv2.circle(kernel, center, radius, 1, thickness=-1)
        
        for i in range(batch_size):
            np_array = tensor[i, 4].cpu().numpy()
            np_array = (np_array * 255).astype(np.uint8)

            if random.random() < 0.5:
                operation = "dilate"
            else:
                operation = "erode"

            if operation == 'dilate':
                processed_np_array = cv2.dilate(np_array, kernel, iterations=self.iterations)
            elif operation == 'erode':
                processed_np_array = cv2.erode(np_array, kernel, iterations=self.iterations)
            
            processed_tensor = torch.from_numpy(processed_np_array.astype(np.float32) / 255).to(tensor.device)
            tensor[i, 4] = processed_tensor
        
        return tensor.to("cuda:1")
    
class ZeroMask:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, tensor):
        if tensor.shape[1] != 5:
            raise ValueError("The input tensor must have 5 channels")
        
        batch_size = tensor.shape[0]
        for i in range(batch_size):
            if random.random() < self.p:
                tensor[i, 4] = torch.zeros_like(tensor[i, 4])
        
        return tensor.to("cuda:1")

class OnesMask:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, tensor):
        if tensor.shape[1] != 5:
            raise ValueError("The input tensor must have 5 channels")
        
        batch_size = tensor.shape[0]
        for i in range(batch_size):
            if random.random() < self.p:
                tensor[i, 4] = torch.ones_like(tensor[i, 4])
        
        return tensor.to("cuda:1")