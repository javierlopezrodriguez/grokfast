import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Autoencoder based on: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial9/AE_CIFAR10.html

class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object,):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image.
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2), # 28x28 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 14x14 => 7x7
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Flatten(), # Image grid to single feature vector
            nn.Linear(2*c_hid*7*7, latent_dim) # flattened 7x7, 2*c_hid channels
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels : int,
                 base_channel_size : int,
                 latent_dim : int,
                 act_fn : object,
                 output_act_fn: object | None,):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct.
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
            - output_act_fn : Activation function used at the end of the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2*c_hid*7*7), # flattened 7x7, 2*c_hid channels
            act_fn()
        )
        self.net = nn.Sequential(
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 7x7 => 14x14
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2), # 14x14 => 28x28
            #nn.Tanh() # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )
        if output_act_fn is not None:
            self.net.append(output_act_fn()) # nn.Tanh for (-1,1); nn.Sigmoid for (0,1); nn.ReLU for (0,inf)...

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 7, 7) # flattened 7x7, 2*c_hid channels
        x = self.net(x)
        return x

class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class : object = Encoder,
                 encoder_act_fn: object = nn.ReLU,
                 decoder_class : object = Decoder,
                 decoder_act_fn: object = nn.ReLU,
                 decoder_output_act_fn: object = nn.Sigmoid,
                 num_input_channels: int = 1,
                 width: int = 28,
                 height: int = 28):
        super().__init__()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim, encoder_act_fn)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim, decoder_act_fn, decoder_output_act_fn)

        # Example input array needed for visualizing the graph of the network
        self.example_input = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat # output
    
    def forward_with_latent(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat, z # output, latent
    
# the previous one with the training configuration does not produce anything useful

# based on: https://www.tensorflow.org/tutorials/generative/autoencoder?hl=es-419
class DenoiseAutoencoder(nn.Module):
    def __init__(self, activation_fn: object = nn.ReLU, final_activation_fn: object = nn.Sigmoid):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1, stride=2), # (1,28,28) -> (16,14,14)
            activation_fn(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), padding=1, stride=2), # (16,14,14) -> (8,7,7)
            activation_fn(),
            nn.Conv2d(8, 1, kernel_size=(3, 3), padding="same", stride=1), # (8,7,7) -> (1,7,7) # latent 1x7x7 = 49
            activation_fn(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1, 8, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), # (1,7,7) -> (8,14,14)
            activation_fn(),
            nn.ConvTranspose2d(8, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), # (8,14,14) -> (16,28,28)
            activation_fn(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=1, padding='same'), # (16,28,28) -> (1,28,28)
            final_activation_fn(),
        )

        # Example input array needed for visualizing the graph of the network
        self.example_input = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat # output
    
    def forward_with_latent(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat, z.reshape(x.shape[0], -1) # output, latent (flattened)

# Autoencoder that increases in capacity instead of decreasing (overcomplete)
class DenoiseAutoencoderOvercomplete(nn.Module):
    def __init__(self, activation_fn: object = nn.ReLU, final_activation_fn: object = nn.Sigmoid):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, stride=2), # (1,28,28) -> (32,14,14)
            activation_fn(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, stride=2), # (32,14,14) -> (64,7,7)
            activation_fn(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), # (64,7,7) -> (32,14,14)
            activation_fn(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1), # (32,14,14) -> (16,28,28)
            activation_fn(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), stride=1, padding='same'), # (16,28,28) -> (1,28,28)
            final_activation_fn(),
        )

        # Example input array needed for visualizing the graph of the network
        self.example_input = torch.zeros(2, 1, 28, 28)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat # output
    
    def forward_with_latent(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x) # latent representation
        x_hat = self.decoder(z) # output (reconstructed input)
        return x_hat, z.reshape(x.shape[0], -1) # output, latent (flattened)

if __name__ == "__main__":
    # Define a hook to capture and print the shape of tensors
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: Input Shape: {input[0].shape} -> Output Shape: {output.shape}")

    """
    # 28 * 28 = 784 - latent dim of 64 is a good compression maybe?
    conv_ae = Autoencoder(base_channel_size=32, latent_dim=64)

    # Register hooks to each layer in the encoder and decoder
    for layer in conv_ae.encoder.net + conv_ae.decoder.linear + conv_ae.decoder.net:
        layer.register_forward_hook(hook_fn)
    """

    conv_ae = DenoiseAutoencoderOvercomplete()
    for layer in conv_ae.encoder + conv_ae.decoder:
        layer.register_forward_hook(hook_fn)

    loss_fn = nn.MSELoss(reduction="sum")
    from torch.optim import AdamW
    opt = AdamW(conv_ae.parameters())
    print("---")
    print("Model forward")
    print("---")
    reconstructed, latent = conv_ae.forward_with_latent(conv_ae.example_input)
    print("Input", conv_ae.example_input.shape)
    print("Reconstruction", reconstructed.shape) 
    print("Latent", latent.shape)
    opt.zero_grad()
    loss = loss_fn(reconstructed, conv_ae.example_input)
    loss.backward()
    print("---")
    print("Parameters")
    print("---")
    grads = {n: p.grad.data.detach() for n, p in conv_ae.named_parameters() if p.requires_grad and p.grad is not None}
    for k, v in grads.items():
        print(k, ":", v.shape, f"(Total: {v.numel()})")
    print("Total number of parameters:", sum([v.numel() for v in grads.values()]))
    