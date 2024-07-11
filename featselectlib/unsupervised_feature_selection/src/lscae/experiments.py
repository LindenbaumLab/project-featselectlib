import torch
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms
from pathlib import Path
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from scipy.stats import uniform
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import numpy as np
from . import Lscae
from . import gated_laplacian as gl

_all__ = [
    'setup_model', 'create_unsupervised_dataloaders', 'create_twomoon_dataset','correct_feats_selection'
]

def setup_model(input_dim,num_epochs=300, model_type='lscae'): 
    cfg = OmegaConf.create({
        "input_dim": input_dim,          # Dimension of input dataset (total #features)
        "k_selected": 2,            # Number of selected features
        "decoder_lr": 1e-3,         # Decoder learning rate
        "selector_lr": 1e-1,        # Concrete layer learning rate
        "min_lr": 1e-5,             # Minimal layer learning rate
        "weight_decay": 0,          # l2 weight penalty
        "batch_size": 64,           # Minibatch size
        "hidden_dim": 128,          # Hidden layers size
        "model": model_type,        # lscae | cae | ls | gated_laplacian
        "scale_k": 2,               # Number of neighbors for computation of local scales for the kernel
        "laplacian_k": 50,          # Number of neighbors of each pooint, used for computation of the Laplacian
        "start_temp": 10,           # Initial temperature
        "min_temp": 1e-2,           # Final temperature
        "rec_lambda": .5,           # Balance between reconstruction and LS terms
        "num_epochs": num_epochs,          # Number of training epochs
        "verbose": True             # Whether to print to console during training
    })
    model = Lscae(input_dim=cfg.input_dim, kwargs=cfg)
    return model, cfg

def create_unsupervised_dataloaders(batch_size=64, download=True):
    # Ensure transform is applied (if provided)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts images to tensor format and scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,)),  # Normalizes using the global mean and std
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the images from 28x28 to 784
    ])

    # Load MNIST dataset in unsupervised mode using train=True and target_transform=None
    train_dataset = MNIST(root='./data', train=True, download=download, transform=transform, target_transform=None)
    test_dataset = MNIST(root='./data', train=False, download=download, transform=transform, target_transform=None)

    # Separate input images and labels (if labels exist, even though unused)
    train_images = train_dataset.data.float()  # Convert to float
    test_images = test_dataset.data.float()

    # Create Dataloaders
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, shuffle=False)

    # Return loaders, flattened input dimension, and (optional) labels
    input_dim = (train_images.shape[1] * train_images.shape[2],)  # Get flattened input dimension (number of pixels)
    labels = None  # Since labels are unused in unsupervised learning, set to None for clarity
    return train_loader, test_loader, input_dim, labels

def create_twomoon_dataset(n=1200, d=10, noise=0.1):
    """
    Creates two moon clusters in 2D, adding p nuisance features and d noisy copies of one of the original features
    n: size of data (int)
    d: number of nuisance dimensions (int), and number of redundant copies
    noise: noise level (double)
    """
    relevant, y = make_moons(n_samples=n, shuffle=True, noise=noise, random_state=None)
    nuisance = uniform.rvs(size=[n, d])
    data = np.concatenate([relevant, nuisance], axis=1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return data


def correct_feats_selection(trials=10, max_nuisance=15, step=3, selected_model='All',device=None):
    device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running experiments on:", device)
    
    model_names = ['lscae', 'cae', 'ls','gated_laplacain']
    if selected_model != 'All' and selected_model in model_names:
        model_names = [selected_model]
    
    num_models = len(model_names)
    accuracies = np.zeros((int((max_nuisance-3)/step + 1), num_models))
    
    for i, nuisance_dim in enumerate(range(3, max_nuisance+1, step)):
        acc = np.zeros(num_models)
        for _ in range(trials):
            X = create_twomoon_dataset(n=100,d=nuisance_dim)
            dataset = data.TensorDataset(torch.Tensor(X))
            loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

            results = {}
            for idx, name in enumerate(model_names):
                num_epochs=1000
                if name=='gated_laplacain':
                    num_epochs=5000
                    if nuisance_dim>7:
                        num_epochs=12000
                    model=gl.GatedLaplacianModel(input_dim=X.shape[1], seed=1, lam=0.1, fac=2, knn=5,
                                is_param_free_loss=True, num_epochs=num_epochs, batch_size=64,
                                learning_rate=0.01)
                    selected_features = model.select_features(loader)
                else:
                    model, _ = setup_model(X.shape[1],num_epochs,name)
                    selected_features, _, _, _ = model.select_features(loader)
                results[name] = set(selected_features)

            correct_features = {0, 1}
            for idx, name in enumerate(model_names):
                selected_features = results[name]
                correct_count = len(correct_features.intersection(selected_features))
                acc[idx] += correct_count / len(correct_features)

        acc /= trials
        accuracies[i, :] = acc

    return accuracies

   



