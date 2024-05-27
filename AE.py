import os
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, act_fn: object = nn.GELU,img_size: int = 128, layers: int = 3, bottleneck: str = "True"):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. 
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           img_size : The height/width of the image. Image has to be a square with one channel
           act_fn : Activation function used throughout the encoder network
           layers : Number of encoding layers
           bottleneck : makes the latent space channel size equal to base_channel_size 
        """
        super().__init__()

        encoder = []
        c_hid = base_channel_size
        for i in range(layers):
            if i == 0:
                encoder.append(nn.Conv2d(num_input_channels, base_channel_size, kernel_size=3, padding=1,stride=2))
                encoder.append(nn.BatchNorm2d(base_channel_size))
            elif i == layers-1 and bottleneck == "True":
                encoder.append(nn.Conv2d(c_hid, base_channel_size, kernel_size=3, padding=1, stride=2))
                encoder.append(nn.BatchNorm2d(base_channel_size))
            else:
                encoder.append(nn.Conv2d(c_hid, c_hid+base_channel_size, kernel_size=3, padding=1, stride=2))
                c_hid = c_hid+base_channel_size
                encoder.append(nn.BatchNorm2d(c_hid))

            encoder.append(act_fn())
        self.net = nn.Sequential(*encoder)
            

    def forward(self, x, label):
        x = self.net(x)
        return x

    
class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, act_fn: object = nn.GELU, img_size: int = 128,layers: int = 3,bottleneck: str = "True"):
        """Decoder.

        Args:
           num_input_channels : Number of input channels of the image. 
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           img_size : The height/width of the image. Image has to be a square with one channel
           act_fn : Activation function used throughout the encoder network
           layers : Number of encoding layers
           bottleneck : makes the latent space channel size equal to base_channel_size 
        """
        super().__init__()
        decoder = []
        c_hid = base_channel_size*layers
        for i in reversed(range(layers)):
            if i == 0:
                decoder.append(nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, padding=1,stride=2,output_padding=1))
            elif i == layers-1 and bottleneck == "True":
                c_hid = c_hid-base_channel_size
                decoder.append(nn.ConvTranspose2d(base_channel_size, c_hid, kernel_size=3, padding=1, stride=2,output_padding=1))
                decoder.append(nn.BatchNorm2d(c_hid))
            else:
                decoder.append(nn.ConvTranspose2d(c_hid, c_hid-base_channel_size, kernel_size=3, padding=1, stride=2,output_padding=1))
                decoder.append(nn.BatchNorm2d(c_hid-base_channel_size))
                c_hid = c_hid-base_channel_size
            decoder.append(act_fn())
        self.net = nn.Sequential(*decoder)

    def forward(self, x, label):
        x = self.net(x)
        return x


class AE(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int = 256,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 1,
        width: int = 128,
        height: int = 128,
        lr: float = 1e-4,
        layers: int = 3,
        bottleneck: str = "True"
    ):
        """AE.

        Args:
           num_input_channels : Number of input channels of the image. 
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           img_size : The height/width of the image. Image has to be a square with one channel
           encoder_class : The encoder class 
           decoder_class : The decoder class 
           act_fn : Activation function used throughout the encoder network
           layers : Number of encoding layers
           lr : learning rate
           bottleneck : makes the latent space channel size equal to base_channel_size 
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size,layers = layers, bottleneck = bottleneck)
        self.decoder = decoder_class(num_input_channels, base_channel_size, layers = layers,bottleneck = bottleneck)
        # Example input array needed for visualizing the graph of the network
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.lr = lr


    
    def forward(self, x, y):
        """The forward function takes in an image and returns the reconstructed image."""
        x = self.encoder(x,y)
        return [self.decoder(x,y)] # list to be same format as AE

    def _get_reconstruction_loss(self, x, x_source, y):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x_hat = self(x,y)
        loss = F.mse_loss(x_source, x_hat[0], reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    
    def training_step(self, batch, batch_idx):
        image,source,label = batch
        loss = self._get_reconstruction_loss(image,source,label)
        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image,source,label = batch
        loss = self._get_reconstruction_loss(image,source,label)
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        image,source,label = batch
        loss = self._get_reconstruction_loss(image,source,label)
        return loss

    def on_train_epoch_end(self):
        epoch_mean  = torch.stack(self.training_step_outputs).mean()
        self.log("train_loss", epoch_mean)
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        epoch_mean  = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss", epoch_mean)
        self.validation_step_outputs.clear()
        

