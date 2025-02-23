import torch
from torchvision import datasets, transforms
from .trainer_dataset import KenyanFood13Dataset, TransformedSubset
from torch.utils.data import Dataset, DataLoader, random_split

def get_mean_std(dataset, batch_size=8, num_workers=4):
    
    # transform = image_preprocess_transforms()
    
    # loader = data_loader(data_root, transform)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    batch_mean = torch.zeros(3)
    batch_mean_sqrd = torch.zeros(3)

    for batch_data, _ in loader:
        batch_mean += batch_data.mean(dim=(0, 2, 3)) # E[batch_i] 
        batch_mean_sqrd += (batch_data ** 2).mean(dim=(0, 2, 3)) #  E[batch_i**2]
    
    # E[dataset] = E[E[batch_1], E[batch_2], ...]
    mean = batch_mean / len(loader)
    
    # var[X] = E[X**2] - E[X]**2
    
    # E[X**2] = E[E[batch_1**2], E[batch_2**2], ...]
    # E[X]**2 = E[E[batch_1], E[batch_2], ...] ** 2
    
    var = (batch_mean_sqrd / len(loader)) - (mean ** 2)
        
    std = var ** 0.5
    # print('mean: {}, std: {}'.format(mean, std))
    
    return mean, std


def get_data(batch_size, data_root='data', num_workers=1):
    compulsary_preprocess = transforms.Compose([
        # Resize to 32X32
        # transforms.Resize((32, 32)),
        # this re-scale image tensor values between 0-1. image_tensor /= 255
        # transforms.ToTensor(),
        # subtract mean (0.1307) and divide by variance (0.3081).
        # This mean and variance is calculated on training data (verify yourself)
        # transforms.Normalize((0.1307, ), (0.3081, ))
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    dataset =  KenyanFood13Dataset(data_root, image_shape=None, transform=compulsary_preprocess)
    classes = dataset.get_classes()
    
    train_size = int(0.8 * len(dataset)) # 80% for training
    test_size = len(dataset) - train_size # 20% for validation

    train_dataset_compulsary_prepocess, test_dataset = random_split(dataset, [train_size, test_size])

    # test dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    train_mean, train_std = get_mean_std(train_dataset_compulsary_prepocess, batch_size=batch_size, num_workers=num_workers)

    train_preprocess = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomCrop(28, padding=4),
        # transforms.PILToTensor(),
        # transforms.ConvertImageDtype(torch.float),
        # transforms.RandomPerspective(distortion_scale=0.6, p=1),
        # transforms.ColorJitter(brightness=.5, hue=.3),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms.Normalize(mean=train_mean, std=train_std)
    ])

    # Apply transformation to the subset
    train_dataset_subset = TransformedSubset(train_dataset_compulsary_prepocess, train_preprocess)

    train_loader = torch.utils.data.DataLoader(
        train_dataset_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader, test_loader, train_mean, train_std, classes