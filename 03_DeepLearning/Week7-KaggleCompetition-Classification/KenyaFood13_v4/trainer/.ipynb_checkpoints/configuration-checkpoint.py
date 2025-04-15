# # <font style="color:blue">Configurations</font>
import os
from typing import Callable, Iterable
from dataclasses import dataclass

from torchvision.transforms import ToTensor

import torch
import torch.nn as nn


# ## <font style="color:green">System Configuration</font>

base_dir = "../../../../data/Week7_project2_classification/"
@dataclass
class SystemConfig:
    seed: int = 42  # seed number to set the state of all random number generators
    cudnn_benchmark_enabled: bool = False  # enable CuDNN benchmark for the sake of performance
    cudnn_deterministic: bool = True  # make cudnn deterministic (reproducible training)


# ## <font style="color:green">Data Configuration</font>

@dataclass
class DatasetConfig:
    # root_dir: str = "data"  # dataset directory root
    root_dir: str = os.path.join(base_dir, "KenyanFood13Dataset")
    train_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during training data preparation
    test_transforms: Iterable[Callable] = (
        ToTensor(),
    )  # data transformation to use during test data preparation


# ## <font style="color:green">Dataloader Configuration</font>

@dataclass
class DataloaderConfig:
    batch_size: int = 5 #250  # amount of data to pass through the network at each forward-backward iteration
    num_workers: int = 5  # number of concurrent processes using to prepare data


# ## <font style="color:green">Optimizer Configuration</font>

@dataclass
class OptimizerConfig:
    learning_rate: float = 0.001  # determines the speed of network's weights update
    momentum: float = 0.9  # used to improve vanilla SGD algorithm and provide better handling of local minimas
    weight_decay: float = 0.0001  # amount of additional regularization on the weights values
    lr_step_milestones: Iterable = (
        30, 40
    )  # at which epoches should we make a "step" in learning rate (i.e. decrease it in some manner)
    lr_gamma: float = 0.1  # multiplier applied to current learning rate at each of lr_ctep_milestones


# ## <font style="color:green">Training Configuration</font>

@dataclass
class TrainerConfig:
    trainer_name: str = "base_trainer"
    model_name_prefix: str = trainer_name + ".pt"
    model_dir: str = os.path.join(base_dir, trainer_name, "checkpoints") # directory to save model states
    tensor_board_dir: str = os.path.join(base_dir, trainer_name, "runs") 
    model_saving_frequency: int = 1  # frequency of model state savings per epochs
    device: str = "cpu"  # device to use for training.
    epoch_num: int = 1 #50  # number of times the whole dataset will be passed through the network
    progress_bar: bool = False  # enable progress bar visualization during train process
    submission_dir: str = os.path.join(base_dir, trainer_name, "submissions")
    
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # convolution layers
        self._body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Fully connected layers
        self._head = nn.Sequential(
            nn.Linear(in_features=64*52*52, out_features=1024), 
            nn.ReLU(inplace=True),
            
            nn.Linear(in_features=1024, out_features=13)
        )
    
    def forward(self, x):        
        # apply feature extractor
        x = self._body(x)
        # flatten the output of conv layers
        # dimension should be batch_size * number_of weight_in_last conv_layer
        x = x.view(x.size()[0], -1)
        # apply classification head
        x = self._head(x)
        
        return x