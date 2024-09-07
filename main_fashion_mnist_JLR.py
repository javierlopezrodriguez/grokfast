import random
import time
import math
from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision
from torchvision import transforms as T

from grokfast import *
#from model import Autoencoder, Encoder, Decoder
#from model import DenoiseAutoencoder
from model import DenoiseAutoencoderOvercomplete

from sklearn.manifold import TSNE


def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50, noisy_input=False, noise_factor=0.2):
    """Computes mean loss of `network` on `dataset`.
    """
    network.eval()
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            x = x.to(device)
            if noisy_input: # https://www.tensorflow.org/tutorials/generative/autoencoder?hl=es-419#second_example_image_denoising
                x_noisy = x + noise_factor * torch.randn(x.shape, device=device)
                x_noisy = torch.clamp(x_noisy, 0, 1)
                x_hat = network(x_noisy) # denoising
            else: # not noisy input
                x_hat = network(x) # reconstruction
            if loss_function == 'MSE':
                total += loss_fn(x_hat, x).item()
            else:
                pass # more losses
            points += len(labels)
        return total / points


optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
}

def plot_losses(log_steps, train_losses, test_losses, loss_function, save_path=None):
    """
    Plots the train and test loss curves and saves the plot as a PNG file.

    Parameters:
    - log_steps: List or array of log steps (x-axis values)
    - train_losses: List or array of training losses (y-axis values for train curve)
    - test_losses: List or array of test losses (y-axis values for test curve)
    - loss_function: String specifying the loss function used (for the ylabel)
    - save_path: File path which the plot will be saved as
    """
    plt.figure()
    # Create the plot
    plt.plot(log_steps, train_losses, label="train")
    plt.plot(log_steps, test_losses, label="val")
    
    # Add plot elements
    plt.legend()
    title = "Fashion MNIST Image Reconstruction"
    plt.title(title)
    plt.xlabel("Optimization Steps")
    plt.ylabel(f"{loss_function} Loss")
    
    # Use log scale for both axes
    plt.xscale("log", base=10)
    plt.yscale("log", base=10)
    plt.grid()
    
    # Save the plot
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

# Function to extract latent vectors and labels
def get_latent_vectors(ae, data_loader, device):
    latent_vectors = []
    labels = []
    ae.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            _, z = ae.forward_with_latent(x)
            latent_vectors.append(z.cpu().numpy())
            labels.append(y.numpy())
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)
    return latent_vectors, labels

# Plotting function for t-SNE
def plot_tsne(latent_2d, labels, title, save_path=None):
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10', s=10)
    plt.colorbar(scatter, label="Class")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

# Plotting function for t-SNE with dataset and class labels
def plot_combined_tsne(latent_2d, split_labels, class_labels, title, save_path=None):
    plt.figure(figsize=(8, 8))
    # Define markers and color map for visualization
    markers = ['o', 'x']  # 'o' for train, 'x' for test

    # Plot train and test points with different markers and colors for class labels
    for i, dataset_label in enumerate(np.unique(split_labels)):  # 0 for train, 1 for test
        indices = split_labels == dataset_label
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1],
                    c=class_labels[indices], cmap='tab10', label=f"{'Train' if dataset_label == 0 else 'Test'}",
                    marker=markers[dataset_label], s=10, alpha=0.7)

    plt.colorbar(label="Class Label")
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.legend(title="Dataset", loc="best")

    # Save figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

# Function to plot original and reconstructed images
def plot_reconstructions(x, x_hat, num_images=7, save_path=None, x_noisy=None):
    """
    Plot original images (x), reconstructed images (x_hat), and optionally noisy images (x_noisy).
    
    Parameters:
    - x: Original images (tensor)
    - x_hat: Reconstructed images (tensor)
    - num_images: Number of images to display (default is 6)
    - save_path: Path to save the plot (default is None, meaning the plot is not saved)
    - x_noisy: Optional noisy images to be plotted in the first row
    """
    # Convert tensors to numpy arrays for plotting
    x = x[:num_images].detach().cpu().numpy()  # Get the first `num_images` images
    x_hat = x_hat[:num_images].detach().cpu().numpy()
    
    rows = 3 if x_noisy is not None else 2  # If noisy images are provided, use 3 rows
    fig, axes = plt.subplots(rows, num_images, figsize=(num_images * 2, rows * 2))
    
    if x_noisy is not None:
        x_noisy = x_noisy[:num_images].detach().cpu().numpy()
        for i in range(num_images):
            # Noisy images on the first row
            axes[0, i].imshow(x_noisy[i].squeeze(), cmap='gray')
            axes[0, i].axis('off')
            if i == num_images // 2:
                axes[0, i].set_title("Noisy")
    
    orig_axis = 1 if x_noisy is not None else 0
    rec_axis = orig_axis + 1

    for i in range(num_images):
        # Original images on the second row (or first if no noisy images)
        axes[orig_axis, i].imshow(x[i].squeeze(), cmap='gray')
        axes[orig_axis, i].axis('off')
        if i == num_images // 2:
            axes[orig_axis, i].set_title("Original")

        # Reconstructed images on the last row
        axes[rec_axis, i].imshow(x_hat[i].squeeze(), cmap='gray')
        axes[rec_axis, i].axis('off')
        if i == num_images // 2:
            axes[rec_axis, i].set_title("Reconstructed")
    
    plt.tight_layout()

    # Save the plot if save_path is specified
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()

def main(args):
    log_freq = math.ceil(args.optimization_steps / 150)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    torch.set_default_dtype(dtype)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # load dataset
    train = torchvision.datasets.FashionMNIST(root=args.download_directory, train=True, 
        transform=T.ToTensor(), download=True) # ToTensor scales to [0,1]
    test = torchvision.datasets.FashionMNIST(root=args.download_directory, train=False, 
        transform=T.ToTensor(), download=True) 
    train = Subset(train, range(args.train_points))
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True)

    assert args.activation in activation_dict, f"Unsupported activation function: {args.activation}"
    activation_fn = activation_dict[args.activation]

    # create model
    """
    ae = Autoencoder(base_channel_size=32,
                     latent_dim=64,
                     encoder_class=Encoder,
                     encoder_act_fn=activation_fn,
                     decoder_class=Decoder,
                     decoder_act_fn=activation_fn,
                     decoder_output_act_fn=nn.Sigmoid, # [0,1] to compare with the scaled input
                     num_input_channels=1,
                     width=28,
                     height=28)
    """
    #ae = DenoiseAutoencoder(activation_fn, nn.Sigmoid)
    ae = DenoiseAutoencoderOvercomplete(activation_fn, nn.Sigmoid)
    ae.to(device)
    with torch.no_grad():
        for p in ae.parameters():
            p.data = args.initialization_scale * p.data
    nparams = sum([p.numel() for p in ae.parameters() if p.requires_grad])
    print(f'Number of parameters: {nparams}')

    # create optimizer
    assert args.optimizer in optimizer_dict, f"Unsupported optimizer choice: {args.optimizer}"
    optimizer = optimizer_dict[args.optimizer](ae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # define loss function
    assert args.loss_function in loss_function_dict
    loss_fn = loss_function_dict[args.loss_function]()

    # noisy input
    noisy_input = args.noise_factor != 0
    if noisy_input:
        print("Noise factor:", args.noise_factor)

    train_losses, test_losses = [], []
    norms, last_layer_norms, log_steps = [], [], []
    grads = None

    steps = 0
    with tqdm(total=args.optimization_steps, dynamic_ncols=True) as pbar:
        for x, labels in islice(cycle(train_loader), args.optimization_steps):
            x = x.to(device)
            do_log = (steps < 30) or (steps < 150 and steps % 10 == 0) or steps % log_freq == 0
            if do_log:
                train_losses.append(compute_loss(ae, train, args.loss_function, device, 
                                                 N=len(train), 
                                                 noisy_input=noisy_input, 
                                                 noise_factor=args.noise_factor))
                test_losses.append(compute_loss(ae, test, args.loss_function, device, 
                                                N=len(test), 
                                                noisy_input=noisy_input, 
                                                noise_factor=args.noise_factor))
                log_steps.append(steps)

                pbar.set_description(
                    "L: {0:1.1e}|{1:1.1e}".format(
                        train_losses[-1],
                        test_losses[-1],
                    )
                )

            ae.train()

            if noisy_input: # https://www.tensorflow.org/tutorials/generative/autoencoder?hl=es-419#second_example_image_denoising
                x_noisy = x + args.noise_factor * torch.randn(x.shape, device=device)
                x_noisy = torch.clamp(x_noisy, 0, 1)
                x_hat = ae(x_noisy) # denoising
            else: # not noisy input
                x_hat = ae(x) # reconstruction
                x_noisy=None

            if args.loss_function == 'MSE':
                loss = loss_fn(x_hat, x)
            else:
                pass # more losses

            optimizer.zero_grad()
            loss.backward()

            #######

            trigger = False

            if args.filter == "none":
                pass
            elif args.filter == "ma":
                grads = gradfilter_ma(ae, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
            elif args.filter == "ema":
                grads = gradfilter_ema(ae, grads=grads, alpha=args.alpha, lamb=args.lamb)
            else:
                raise ValueError(f"Invalid gradient filter type `{args.filter}`")

            #######

            optimizer.step()

            steps += 1
            pbar.update(1)

            if do_log:
                plot_losses(log_steps, train_losses, test_losses, args.loss_function, save_path=f"results/fashion_mnist_{args.label}_loss.png")
                plot_reconstructions(x, x_hat, 7, save_path=f"results/fashion_mnist_{args.label}_images.png", x_noisy=x_noisy)

                torch.save({
                    'its': log_steps,
                    'train_loss': train_losses,
                    'val_loss': test_losses,
                }, f"results/fashion_mnist_{args.label}.pt")

    # Save final model
    torch.save(ae.state_dict(), f"results/fashion_mnist_{args.label}_model_weights.pt")

    # TSNE
    # load dataset (again because we did subset the train earlier)
    train = torchvision.datasets.FashionMNIST(root=args.download_directory, train=True, 
        transform=T.ToTensor(), download=True) # ToTensor scales to [0,1]
    test = torchvision.datasets.FashionMNIST(root=args.download_directory, train=False, 
        transform=T.ToTensor(), download=True) 
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False)
    
    # Extract latent vectors and labels for train and test sets
    train_latent, train_labels = get_latent_vectors(ae, train_loader, device)
    test_latent, test_labels = get_latent_vectors(ae, test_loader, device)

    # Combine train and test latent vectors
    combined_latent = np.concatenate([train_latent, test_latent], axis=0)
    combined_labels = np.concatenate([train_labels, test_labels], axis=0)
    combined_split = np.concatenate([np.zeros_like(train_labels), np.ones_like(test_labels)])  # 0 for train, 1 for test

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=args.seed, verbose=3, n_jobs=-1) # default perplexity etc.

    # Reduce dimensionality separately for train, test, and combined
    #train_latent_2d = tsne.fit_transform(train_latent)
    #test_latent_2d = tsne.fit_transform(test_latent)
    combined_latent_2d = tsne.fit_transform(combined_latent)

    # Plot t-SNE
    train_title = "t-SNE of Train Latent Vectors with Class Labels"
    train_save_path = f"results/fashion_mnist_{args.label}_tsne_train.png"
    test_title = "t-SNE of Test Latent Vectors with Class Labels"
    test_save_path = f"results/fashion_mnist_{args.label}_tsne_test.png"
    plot_tsne(combined_latent_2d[:train_latent.shape[0]], train_labels, train_title, train_save_path)
    plot_tsne(combined_latent_2d[train_latent.shape[0]:], test_labels, test_title, test_save_path)
    combined_title = "t-SNE of Train and Test Latent Vectors with Class Labels"
    combined_save_path = f"results/fashion_mnist_{args.label}_tsne_combined.png"
    plot_combined_tsne(combined_latent_2d, combined_split, combined_labels, combined_title, combined_save_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--label", default="")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_points", type=int, default=1000)
    parser.add_argument("--optimization_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--loss_function", type=str, default="MSE")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--initialization_scale", type=float, default=8.0)
    parser.add_argument("--download_directory", type=str, default=".")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width", type=int, default=200)
    parser.add_argument("--activation", type=str, default="ReLU")

    # Task
    parser.add_argument("--noise_factor", type=float, default=0.0)

    # Grokfast
    parser.add_argument("--filter", type=str, choices=["none", "ma", "ema", "fir"], default="none")
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--lamb", type=float, default=5.0)
    args = parser.parse_args()

    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{args.lamb:.2f}'.replace('.', '')

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    noise_suffix = ''
    if args.noise_factor != 0:
        noise_suffix += f'_noisy{args.noise_factor:.3f}'.replace('.', '')

    args.label = args.label + filter_str + filter_suffix + optim_suffix + noise_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
