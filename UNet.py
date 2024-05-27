import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim

from UNet_layers import *

class UNet(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int = 64, #64 default 
        num_input_channels: int = 1,
        width: int = 128,
        height: int = 128,
        lr: float = 1e-4,
        bilinear: str = "True"
    ):
        """UNet.

        Args:
            num_input_channels : Number of input channels of the image. 
            base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            height: The height of the image.
            width : The weidth of the image.
            lr : learning rate
            bilinear : if the decoder should be bilinear or use transposed convolutional 
        """
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        # Example input array needed for visualizing the graph of the network
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.lr = lr
        self.num_input_channels = num_input_channels
        self.bilinear = bilinear


        self.inc = DoubleConv(self.num_input_channels, base_channel_size)
        self.down1 = Down(base_channel_size, base_channel_size*2)
        self.down2 = Down(base_channel_size*2, base_channel_size*4)
        self.down3 = Down(base_channel_size*4, base_channel_size*8)
        factor = 2 if bilinear == "True" else 1
        self.down4 = Down(base_channel_size*8, base_channel_size*16 // factor)
        self.up1 = Up(base_channel_size*16, base_channel_size*8 // factor, bilinear)
        self.up2 = Up(base_channel_size*8, base_channel_size*4 // factor, bilinear)
        self.up3 = Up(base_channel_size*4, base_channel_size*2 // factor, bilinear)
        self.up4 = Up(base_channel_size*2, base_channel_size, bilinear)
        self.outc = OutConv(base_channel_size, num_input_channels)


    
    def forward(self, x, y):
        """The forward function takes in an image and returns the reconstructed image."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return [logits]

    def _get_reconstruction_loss(self, x, x_source, y):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)."""
        x_hat = self(x,y)
        loss = F.mse_loss(x_source, x_hat[0], reduction="none")
        loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        # Using a scheduler is optional but can be helpful.
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