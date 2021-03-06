from typing import Optional
from cv2 import transform
import pytorch_lightning as pl
import torch
from torch import save
from torch.utils import data
from torch.utils.data import random_split, DataLoader
from torch.nn import functional as F
import tonic
import torchvision
from torchvision import transforms
import os
import numpy as np

from project.datamodules.cifar10dvs import CIFAR10DVS
from einops import rearrange
from project.datamodules.ncars import NCARS
from project.utils.dvs_noises import BackgroundActivityNoise, CenteredOcclusion, RandomTimeReversal, background_activity, hot_pixels

# from project.utils.phase_dvs import ToBitEncoding, ToWeightedFrames, ToTimeSurfaceCustom


class DVSDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, dataset: str, data_dir: str = "data/", event_representation: str = "HOTS", timesteps=10, noise: str = None, severity=1, num_workers: int = 0, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.dataset = dataset  # name of the dataset
        self.timesteps = timesteps
        self.noise = noise
        self.severity = severity
        self.event_representation = event_representation
        self.num_workers = num_workers

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

        # transform
        self.sensor_size, self.num_classes = self._get_dataset_info()
        self.train_transform, self.val_transform = self._get_transforms(event_representation)

    def _get_dataset_info(self):
        if self.dataset == "n-mnist":
            return tonic.datasets.NMNIST.sensor_size, len(tonic.datasets.NMNIST.classes)
        elif self.dataset == "cifar10-dvs":
            return CIFAR10DVS.sensor_size, 10
        elif self.dataset == "dvsgesture":
            return tonic.datasets.DVSGesture.sensor_size, len(tonic.datasets.DVSGesture.classes)
        elif self.dataset == "n-caltech101":
            return (224, 224, 2), 101
        elif self.dataset == "asl-dvs":
            return tonic.datasets.ASLDVS.sensor_size, len(tonic.datasets.ASLDVS.classes)
        elif self.dataset == 'ncars':
            return NCARS.sensor_size, len(NCARS.classes)

    def prepare_data(self) -> None:
        # downloads the dataset if it does not exist
        # NOTE: since we use the library named "Tonic", all the download process is handled, we just have to make an instanciation
        if self.dataset == "n-mnist":
            tonic.datasets.NMNIST(save_to=self.data_dir)
        elif self.dataset == "cifar10-dvs":
            CIFAR10DVS(save_to=self.data_dir)
        elif self.dataset == "dvsgesture":
            tonic.datasets.DVSGesture(save_to=self.data_dir)
        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)
        elif self.dataset == "asl-dvs":
            tonic.datasets.ASLDVS(save_to=self.data_dir)
        elif self.dataset == 'ncars':
            NCARS(save_to=self.data_dir, download=True)

    def _get_transforms(self, event_representation: str):
        # denoise = tonic.transforms.Denoise()
        if event_representation == "HATS":
            representation = tonic.transforms.ToAveragedTimesurface(self.sensor_size)
        elif event_representation == "frames_time":
            representation = tonic.transforms.Compose([
                tonic.transforms.ToFrame(self.sensor_size, n_time_bins=self.timesteps),
                transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
                transforms.Lambda(lambda x: rearrange(
                    x, 'frames polarity height width -> (frames polarity) height width'))
            ])
        elif event_representation == "frames_event":
            representation = tonic.transforms.Compose([
                tonic.transforms.ToFrame(self.sensor_size, n_event_bins=self.timesteps),
                transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
                transforms.Lambda(lambda x: rearrange(
                    x, 'frames polarity height width -> (frames polarity) height width'))
            ])

        elif event_representation == "histogram":
            representation = tonic.transforms.Compose([
                tonic.transforms.ToFrame(self.sensor_size, n_event_bins=1),
                # transforms.Lambda(lambda x: (x > 0).astype(np.float32)),
                transforms.Lambda(lambda x: rearrange(
                    x, 'frames polarity height width -> (frames polarity) height width'))
            ])

        if self.noise is not None and self.noise == 'background_activity':
            vt = [
                BackgroundActivityNoise(self.sensor_size, self.severity)
            ]
        if self.noise is not None and self.noise == 'occlusion':
            vt = [
                CenteredOcclusion(self.severity, self.sensor_size)
            ]

        if self.noise is not None and self.noise == 'reverse':
            vt = [
                RandomTimeReversal(p=1.0)
            ]

        if self.noise is None:
            vt = []

        vt.append(representation)
        vt.append(transforms.Lambda(lambda x: x.astype(np.float32)))

        val_transform = tonic.transforms.Compose(vt)

        train_transform = tonic.transforms.Compose([
            # tonic.transforms.RandomTimeReversal(),
            # tonic.transforms.RandomFlipPolarity(),
            # tonic.transforms.RandomFlipLR(self.sensor_size),
            # denoise,
            representation,
            # transforms.Lambda(lambda x: F.upsample(torch.from_numpy(x), size=(224, 224), mode='nearest').numpy()),
            transforms.Lambda(lambda x: x.astype(np.float32))
        ])

        return train_transform, val_transform

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset == "n-mnist":
            self.train_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.NMNIST(
                save_to=self.data_dir, transform=self.val_transform, target_transform=None, train=False)

        elif self.dataset == "cifar10-dvs":
            dataset_train = CIFAR10DVS(save_to=self.data_dir, transform=self.train_transform, target_transform=None)
            dataset_val = CIFAR10DVS(save_to=self.data_dir, transform=self.val_transform, target_transform=None)
            print(len(dataset_train))
            self.train_set, _ = random_split(dataset_train, lengths=[8000, 10000 - 8000])
            _, self.val_set = random_split(dataset_val, lengths=[8000, 10000 - 8000])

        elif self.dataset == "dvsgesture":
            self.train_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.train_transform, target_transform=None, train=True)
            self.val_set = tonic.datasets.DVSGesture(
                save_to=self.data_dir, transform=self.val_transform, target_transform=None, train=False)

        elif self.dataset == "n-caltech101":
            tonic.datasets.NCALTECH101(save_to=self.data_dir)

        elif self.dataset == "asl-dvs":
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.train_transform)
            full_length = len(dataset)
            print(full_length, 0.8 * full_length)
            self.train_set, _ = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])
            dataset = tonic.datasets.ASLDVS(save_to=self.data_dir, transform=self.val_transform)
            _, self.val_set = random_split(dataset, [0.8 * full_length, full_length - (0.8 * full_length)])
        elif self.dataset == 'ncars':
            self.train_set = NCARS(self.data_dir, train=True, transform=self.train_transform)
            self.val_set = NCARS(self.data_dir, train=False, transform=self.val_transform)
            print(len(self.train_set), len(self.val_set))

        print(len(self.train_set), len(self.val_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
