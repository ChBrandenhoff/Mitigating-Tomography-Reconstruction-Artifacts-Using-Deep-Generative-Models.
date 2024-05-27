from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
from yeast_data import *
import random


class Dataset(Dataset):
    def __init__(self, num_samples_per_epoch = 160, h:int = 128, w: int  = 128):
        """Dataset. 
        Creates new data for each epoch. Used for training on always new data. 
        Makes training much slower, as it always generates new data per "__getitem__"

        Args:
            num_samples_per_epoch : number of samples per epoch. 
            h : height of the image
            w : weidth of the image
        """
        self.num_samples_per_epoch = num_samples_per_epoch
        self.h = h
        self.w = w
        self.data = None  # Your data generation mechanism goes here

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self,idx):
        """
        Get a batch of randomly uniformly normalized generated images, where image is a reconstruction of angles given by the label and the original images before reconstruction. 

        return: image, original image, label (theta)
        """
        raw_data = DataLoaderYeast()
        angles_options = [[120,1],[130,1],[140,1],[150,1],[160,1],[170,1],[180,1]]
        angles = [random.choice(angles_options)]
        raw_data.generate_data(w=self.w,h=self.h, noise=0, repeat=1,angles=angles)
        for item in raw_data.data:
            image = np.array(item[0])
            source_image = np.array(item[1])
            label = np.array(item[2])
        min_val, max_val = self.compute_min_max(image, source_image)

        # Convert image to tensor if not already in tensor format
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add a channel dimension (assuming images are grayscale)
        # Normalize the image to [0, 1]
        image = (image - min_val) / (max_val - min_val)
        source_image = torch.tensor(source_image, dtype=torch.float32)

        source_image = source_image.unsqueeze(0)
        source_image = (source_image - min_val) / (max_val - min_val)
        # Convert label to tensor if not already in tensor format
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)  # Add a dimension to match the image shape
        return image,source_image, label
    
    def compute_min_max(self, images, source_images):
        images_concatenated = torch.cat([torch.tensor(image, dtype=torch.float32).unsqueeze(0) for image in images], dim=0)
        source_image_images_concatenated = torch.cat([torch.tensor(image, dtype=torch.float32).unsqueeze(0) for image in source_images], dim=0)
        min_val = torch.min(images_concatenated)
        min_val_source = torch.min(source_image_images_concatenated)
        max_val = torch.max(images_concatenated)
        max_val_source= torch.max(source_image_images_concatenated)
        return torch.min(min_val,min_val_source), torch.max(max_val,max_val_source)
    
class Data_handler():
    def __init__(self, batch_size: int = 16,validation_split: float = .2, shuffle_dataset: bool = True,random_seed = None,file_name: str = "overfitting_120",num_workers=0):
        """Data_handler. 
        Given a dataset, it creates the traing data and validation data

        Args:
            batch_size : size of the batch
            validation_split : the amount of data saved for validation
            shuffle_dataset : if the dataset should be shuffeled
            random_seed : seed the dataset for reproducability
            file_name : name of the file to load
            num_workers : used by pytorch for optimization
        """
        self.raw_data = DataLoaderYeast()
        self.raw_data.load_data(file_name=file_name)
        self.dataset = Dataset_load(self.raw_data.data)
        # Creating data indices for training and validation splits:
        self.dataset_size = len(self.dataset)
        indices = list(range(self.dataset_size))
        split = int(np.floor(validation_split * self.dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, #num_workers=59,
                                                sampler=train_sampler)
        self.validation_loader = DataLoader(self.dataset, batch_size=batch_size,#num_workers=59,
                                                        sampler=valid_sampler)
class Dataset_load(Dataset):
    def __init__(self, data):
        """Dataset_load. 
        Given the Data_handler's training or validation loader, normalize and handles the data for traning 

        Args:
            data : data from Data_handler()
        """
        self.images = data[:, 0] 
        self.source_image = data[:, 1]
        self.labels = data[:, 2]  
        self.min_val, self.max_val = self.compute_min_max()
        

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a batch of images, where the image is a reconstruction of angles given by the label and the original images before reconstruction. 

        return: image, original image, label (theta)
        """
        image = self.images[idx]  # Get image at index 'idx'
        source_image = self.source_image[idx]
        label = self.labels[idx]  # Get label at index 'idx'


        # Convert image to tensor if not already in tensor format
        image = torch.tensor(image, dtype=torch.float32)
        image = image.unsqueeze(0)  # Add a channel dimension (assuming images are grayscale)

        # Normalize the image to [0, 1]
        image = (image - self.min_val) / (self.max_val - self.min_val)
        source_image = torch.tensor(source_image, dtype=torch.float32)

        source_image = source_image.unsqueeze(0)
        source_image = (source_image - self.min_val) / (self.max_val - self.min_val)
        # Convert label to tensor if not already in tensor format
        label = torch.tensor(label, dtype=torch.float32)
        label = label.unsqueeze(0)  # Add a dimension to match the image shape
        return image,source_image, label

    def compute_min_max(self):
        images_concatenated = torch.cat([torch.tensor(image, dtype=torch.float32).unsqueeze(0) for image in self.images], dim=0)
        source_image_images_concatenated = torch.cat([torch.tensor(image, dtype=torch.float32).unsqueeze(0) for image in self.source_image], dim=0)
        min_val = torch.min(images_concatenated)
        min_val_source = torch.min(source_image_images_concatenated)
        max_val = torch.max(images_concatenated)
        max_val_source= torch.max(source_image_images_concatenated)
        return torch.min(min_val,min_val_source), torch.max(max_val,max_val_source)