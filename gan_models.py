import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader



class Generator(nn.Module):
    """Generator Class

    Args:
        input_dim (int): the dimension of the input vector
        im_chan (int): the number of channels of the output image
        hidden_dim (int): the inner dimension of the
    """
    def __init__(self, input_dim=10, im_chan=3, hidden_dim=64):
        # super(Generator, self).__init__()
        super().__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4, kernel_size=4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim, kernel_size=4),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=2, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """Method to return a sequence of operations corresponding to a agenerator block of DCGAN
        
        Args:
            input_channels (int): How many channels the input feature representation has
            output_channels (int): How many channels the output feature representation should have
            kernel_size (int, optional): the size of each convolutional filter
            stride (int, optional): the stride of the convolution
            final_layer (bool, optional): Boolean to affect Activation and BatchNorm
        """
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        """Method for completing a forward pass of the generator: Given an noise tensor

        Args:
            noise (tensor): noise tensor with dimensions (n_samples, input_dim)
        """
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)


def get_noise(n_samples, input_dim, device="cpu"):
    """Function for creating noise vectors from the normal distribution

    Args:
        n_samples (int): the number of samples to generate
        input_dim (int): the dimenstion of the input vector
        device (str, optional): device type
    """
    return torch.randn(n_samples, input_dim, device=device)


def combine_vectors(x, y):
    """Function for concatenanting two vectors

    Args:
        x (tensor): tensor with shape (n_samples, ?)
        y (tensor): tensor with shape (n_samples, ?)
    """
    return torch.cat([x, y], 1)


def get_one_hot_labels(labels, n_classes):
    """Function for getting one hot vectors 
    """
    return F.one_hot(labels, n_classes)



class Classifier(nn.Module):
    """Classifier class

    Args:
        im_chan (int): the number of channels of the output image
        n_classes (int): the total number of classes in the dataset
        hidden_dim (int): the inner dimension
    """
    def __init__(self, im_chan, n_classes, hidden_dim=32):
        super().__init__()
        self.disc = nn.Sequential(
            self.make_classifier_block(im_chan, hidden_dim),
            self.make_classifier_block(hidden_dim, hidden_dim * 2),
            self.make_classifier_block(hidden_dim * 2, hidden_dim * 4),
            self.make_classifier_block(hidden_dim * 4, n_classes, final_layer=True),
        )

    def make_classifier_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        """Method to return a sequence of operations corresponding to a classifier block

        Args:
            input_channels (int): how many channels the input feature representation has
            output_channels (int): how many channels the output feature representation should have
            kernel_size (int, optional): convolutional filter
            stride (int, optional): the size of stride
            final_layer (bool, optional): affects activation and batchnorm
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )
    
    def forward(self, image):
        """Method for completing a forward pass of the classifier

        Args:
            image (tensor): flattened image with im_chan channels
        """
        class_pred = self.disc(image)
        return class_pred.view(len(class_pred), -1)


class Discriminator(nn.Module):
    """Discriminator Class

    Args:
        im_chan (int): the number of channels of the output image
        hidden_dim (int): the number of inner dimensions
    """
    def __init__(self, im_chan=3, hidden_dim=64):
        super().__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim, stride=1),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, hidden_dim * 4),
            self.make_disc_block(hidden_dim * 4, 1, final_layer=True),
        )
    
    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        """Method for returning a sequence of operations corresponding to a Discriminator block

        Args:
            input_channels (int): How many channels the input feature representation has
            output_channels (int): How many channels the output feature representation should have
            kernel_size (int, optional): the size of the covolutional filter
            stride (int, optional): the stride of the convolution
            final_layer (bool, optional): affects activation and batchnorm
        """
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride)
            )
    
    def forward(self, image):
        """Method to complete a forward pass of the discriminator given an image tensor

        Args:
            image (tensor): a flattened image tensor wirh dimensionb (im_chan)
        """
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
