from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import SimpleITK as sitk
import os
import torch
from mp.paths import storage_data_path
from torchvision import transforms
import torch.nn.functional as F
from data_prep.diffusion_utils.diffusion_model_2D import SimpleUnet

import matplotlib.pyplot as plt


class DataGeneratorDiffusionModel:
    def __init__(self, subset, config):
        self.posterior_variance = None
        self.sqrt_one_minus_alphas_cumprod = None
        self.sqrt_recip_alphas = None
        self.beta = None
        self.betas = None
        self.device = config['device']
        self.batch_size = config['batch_szie']
        self.model = SimpleUnet().to(self.device)
        self.model = torch.load(subset)
        self.T = 300
        self.init_hyper()

    def init_hyper(self):
        def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
            return torch.linspace(start, end, timesteps)

        self.beta = linear_beta_schedule(self.T)
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def get_data(self):
        img = torch.randn((self.batch_size, 1, 192, 192)).to(self.device)
        for i in range(0, self.T)[::-1]:
            t = torch.full((1,), i, device=self.device, dtype=torch.long)
            img = self.sample_timestep(img, t)
        return img

    def sample_timestep(self, x, t):
        betas_t = self.get_index_from_list(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = self.get_index_from_list(self.sqrt_recip_alphas, t, x.shape)

        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = self.get_index_from_list(self.posterior_variance, t, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def get_index_from_list(self, vals, t, x_shape):
        """
        Returns a specific index t of a passed list of values vals
        while considering the batch dimension.
        """
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class PredictionDataset(Dataset):
    def __init__(self, subset, size=(192, 192)):
        self.size = size

        dataset_path = os.path.join(storage_data_path, subset)

        # Fetch all patient/study names
        if subset == 'DecathlonHippocampus':
            dataset_path = os.path.join(dataset_path, 'Merged Labels')
            study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                              in os.listdir(dataset_path))
        elif subset == 'DryadHippocampus':
            dataset_path = os.path.join(dataset_path, 'Merged Labels', 'Modality[T1w]Resolution[Standard]')
            study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                              in os.listdir(dataset_path))
        elif subset == 'HarP':
            study_names = set(os.path.join('Training', file_name.split('.nii')[0].split('_gt')[0]) for file_name in
                              os.listdir(os.path.join(dataset_path, 'Training'))) | \
                          set(os.path.join('Validation', file_name.split('.nii')[0].split('_gt')[0]) for file_name in
                              os.listdir(os.path.join(dataset_path, 'Validation')))
        else:
            study_names = set(file_name.split('.nii')[0].split('_gt')[0] for file_name
                              in os.listdir(dataset_path))

        study_names = list(study_names)
        self.img_path = [os.path.join(dataset_path, name + '.nii.gz') for name in study_names]

        data_transforms = [
            transforms.Resize(self.size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            transforms.Lambda(lambda t: (t * 2) - 1)  # Scale between [-1, 1]
        ]
        self.transform = transforms.Compose(data_transforms)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_image = sitk.ReadImage(self.img_path[idx])
        img_image = sitk.GetArrayFromImage(img_image)
        index_n = np.random.randint(0, img_image.shape[0])
        img_image = img_image[index_n]
        img_image = Image.fromarray(np.uint8(255. * img_image))
        if self.transform:
            img_image = self.transform(img_image)
        return img_image


class SimpleDataset(Dataset):
    def __init__(self, subset, size=(192, 192)):
        self.size = size

        dataset_path = os.path.join('/home/zhuzhanshi/MedicalCL/storage/data/PseudoData', subset, 'inference')

        self.img_path = [os.path.join(dataset_path, name) for name in os.listdir(dataset_path)]

        data_transforms = [
            transforms.Resize(self.size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # Scales data into [0,1]
            # transforms.Lambda(lambda t: (t.numpy()-np.mean(t.numpy()))/np.std(t.numpy()))
        ]
        self.transform = transforms.Compose(data_transforms)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_image = Image.open(self.img_path[idx])
        if self.transform:
            img_image = self.transform(img_image)
        return img_image


if __name__ == '__main__':
    train_dataset = SimpleDataset('I2CVB')
    train_loader = DataLoader(train_dataset, batch_size=10, num_workers=8, shuffle=False)
    for batch, prediction_image in enumerate(train_loader):
        print(prediction_image.shape)
        image = np.array(prediction_image[0][0])
        print(np.max(image), np.min(image))
        print(np.average(image), np.std(image))
        plt.imshow(image)
        plt.show()

        break
