import os
import glob
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import requests
import zipfile
import tarfile
from tqdm import tqdm
import shutil

class SRDatasetDownloader:
    """Download and extract standard super-resolution benchmark datasets"""
    
    DATASET_URLS = {
        'Set5': 'https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip',
        'Set14': 'https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip',
        'BSD100': 'https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip',
        'Urban100': 'https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip',
    }
    
    def __init__(self, data_dir='datasets'):
        """
        Args:
            data_dir: Directory to save the datasets
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def download_file(self, url, destination):
        """Download a file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        
        with open(destination, 'wb') as f, tqdm(
                desc=os.path.basename(destination),
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
            for data in response.iter_content(block_size):
                f.write(data)
                progress_bar.update(len(data))
    
    def extract_file(self, file_path, extract_dir):
        """Extract zip or tar file"""
        if file_path.endswith('.zip'):
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
    
    def download_dataset(self, dataset_name):
        """Download and extract dataset if it doesn't exist"""
        if dataset_name not in self.DATASET_URLS:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(self.DATASET_URLS.keys())}")
        
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        
        # Check if dataset already exists
        if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
            print(f"{dataset_name} dataset already exists.")
            return dataset_dir
        
        # Create dataset directory
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Download dataset
        url = self.DATASET_URLS[dataset_name]
        file_name = url.split('/')[-1]
        download_path = os.path.join(self.data_dir, file_name)
        
        print(f"Downloading {dataset_name}...")
        self.download_file(url, download_path)
        
        # Extract dataset
        print(f"Extracting {dataset_name}...")
        self.extract_file(download_path, dataset_dir)
        
        # Clean up downloaded zip file
        os.remove(download_path)
        
        print(f"{dataset_name} dataset ready.")
        return dataset_dir

class SuperResolutionDataset(Dataset):
    """Dataset for super-resolution training"""
    
    def __init__(self, dataset_dirs, patch_size=96, scale_factor=4, 
                 augment=True, split='train', lr_only=False):
        """
        Args:
            dataset_dirs: List of directories containing HR images
            patch_size: Size of cropped HR patches
            scale_factor: Downsampling factor for LR images
            augment: Whether to apply data augmentation
            split: 'train' or 'val'
            lr_only: If True, only returns LR images (for inference)
        """
        self.dataset_dirs = dataset_dirs if isinstance(dataset_dirs, list) else [dataset_dirs]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.split = split
        self.lr_only = lr_only
        
        self.lr_patch_size = patch_size // scale_factor
        
        # Collect all image paths
        self.image_paths = []
        for dataset_dir in self.dataset_dirs:
            self.image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', '*.png'), recursive=True))
            self.image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', '*.jpg'), recursive=True))
            self.image_paths.extend(glob.glob(os.path.join(dataset_dir, '**', '*.bmp'), recursive=True))
        
        if not self.image_paths:
            raise RuntimeError(f"No images found in {self.dataset_dirs}")
        
        print(f"Found {len(self.image_paths)} images for {split}")
        
        # Basic transforms
        self.to_tensor = transforms.ToTensor()
        
        # Bicubic downscaling
        self.bicubic_downscale = lambda x: TF.resize(
            x, 
            size=[x.height // scale_factor, x.width // scale_factor], 
            interpolation=Image.BICUBIC
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        hr_img = Image.open(self.image_paths[idx]).convert('RGB')
        
        # For small validation images, don't crop
        if self.split == 'val':
            # Make sure the dimensions are divisible by the scale factor
            h, w = hr_img.height, hr_img.width
            h = h - (h % self.scale_factor)
            w = w - (w % self.scale_factor)
            hr_img = hr_img.resize((w, h))
            
            # Downscale HR to get LR
            lr_img = self.bicubic_downscale(hr_img)
            
            # Convert to tensors
            lr_tensor = self.to_tensor(lr_img)
            hr_tensor = self.to_tensor(hr_img)
            
            if self.lr_only:
                return lr_tensor
            return {'lr': lr_tensor, 'hr': hr_tensor}
        
        # For training, use random crops and augmentations
        if hr_img.height < self.patch_size or hr_img.width < self.patch_size:
            # Resize small images to have at least patch_size dimensions
            ratio = max(self.patch_size / hr_img.height, self.patch_size / hr_img.width)
            new_h = int(hr_img.height * ratio)
            new_w = int(hr_img.width * ratio)
            hr_img = hr_img.resize((new_w, new_h), Image.BICUBIC)
        
        # Random crop
        h, w = hr_img.height, hr_img.width
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        hr_patch = hr_img.crop((left, top, left + self.patch_size, top + self.patch_size))
        
        # Downscale HR patch to get LR patch
        lr_patch = self.bicubic_downscale(hr_patch)
        
        # Apply augmentations
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                hr_patch = TF.hflip(hr_patch)
                lr_patch = TF.hflip(lr_patch)
            
            # Random vertical flip
            if random.random() > 0.5:
                hr_patch = TF.vflip(hr_patch)
                lr_patch = TF.vflip(lr_patch)
            
            # Random 90-degree rotation
            rotation_times = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
            if rotation_times > 0:
                hr_patch = TF.rotate(hr_patch, 90 * rotation_times)
                lr_patch = TF.rotate(lr_patch, 90 * rotation_times)
        
        # Convert to tensors
        lr_tensor = self.to_tensor(lr_patch)
        hr_tensor = self.to_tensor(hr_patch)
        
        if self.lr_only:
            return lr_tensor
        return {'lr': lr_tensor, 'hr': hr_tensor}

def get_sr_datasets(data_dir='datasets', patch_size=96, scale_factor=4, batch_size=16, num_workers=4):
    """
    Setup Super Resolution datasets and dataloaders for all benchmark datasets
    
    Args:
        data_dir: Directory to save/load datasets
        patch_size: Size of HR patches for training
        scale_factor: Downsampling factor
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        
    Returns:
        dict: Dictionary containing train and validation dataloaders
    """
    # Initialize dataset downloader
    downloader = SRDatasetDownloader(data_dir)
    
    # Download datasets if they don't exist
    datasets = {}
    for dataset_name in ['Set5', 'Set14', 'BSD100', 'Urban100']:
        dataset_dir = downloader.download_dataset(dataset_name)
        datasets[dataset_name] = dataset_dir
    
    # Use BSD100 for training, others for validation
    train_dataset = SuperResolutionDataset(
        datasets['BSD100'], 
        patch_size=patch_size, 
        scale_factor=scale_factor, 
        augment=True, 
        split='train'
    )
    
    val_datasets = {}
    for val_dataset_name in ['Set5', 'Set14', 'Urban100']:
        val_datasets[val_dataset_name] = SuperResolutionDataset(
            datasets[val_dataset_name],
            patch_size=patch_size,
            scale_factor=scale_factor,
            augment=False,
            split='val'
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loaders = {}
    for val_name, val_dataset in val_datasets.items():
        val_loaders[val_name] = DataLoader(
            val_dataset,
            batch_size=1,  # Use batch size 1 for validation
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
    
    return {
        'train_loader': train_loader,
        'val_loaders': val_loaders
    }

def custom_sr_dataset(image_dir, patch_size=96, scale_factor=4, batch_size=16, num_workers=4):
    """
    Create a dataset from a custom directory of images
    
    Args:
        image_dir: Directory containing HR images
        patch_size: Size of HR patches for training
        scale_factor: Downsampling factor
        batch_size: Batch size for training
        num_workers: Number of worker threads for data loading
        
    Returns:
        DataLoader: DataLoader for custom dataset
    """
    custom_dataset = SuperResolutionDataset(
        image_dir,
        patch_size=patch_size,
        scale_factor=scale_factor,
        augment=True,
        split='train'
    )
    
    custom_loader = DataLoader(
        custom_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return custom_loader
