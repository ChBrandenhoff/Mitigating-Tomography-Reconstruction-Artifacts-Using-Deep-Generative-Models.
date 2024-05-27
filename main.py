from AE import *
from UNet import UNet
from data_handler import *
import torch
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import pytorch_lightning as pl
import matplotlib.pyplot as plt
device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Yeast reconstruction using VAE')
parser.add_argument('--pretrained', type=str, default="False", help='path to pretrained model, False makes it not load path(default: False)')
parser.add_argument('--model', type=str, default="AE", help='model name (default: AE)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
parser.add_argument('--batch-size', type=int, default=16, help='batch size for data (default: 16)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimiser (default: 1e-4)')
parser.add_argument('--base-channel-size', type=int, default=32, help='Number of channels in the first convolutional layer (default: 32)')
parser.add_argument('--patience', type=int, default=10, help='patience for early stopping (default: 15)')
parser.add_argument('--load-data', type=str, default="False", help='load data from file if true else generate new data each epoch  (default: False)')
parser.add_argument('--image-size', type=int, default=128, help='size of image  (default: 128)')
parser.add_argument('--plot', type=str, default='noise_256_plot', help='Name of file to plot after training. Takes the first image in the dataset given  (default: noise_256_plot)')

# Parse arguments
args, unknown = parser.parse_known_args()
# model handler
if args.model == "AE":
    parser.add_argument('--layers', type=int, default=3, help='number of convolutional layers in AE (default: 3)')
    parser.add_argument('--bottleneck', type=str, default="True", help='True if latent space should be base-channel-size (default: True)')
    parser.add_argument('--filename', type=str, default="test", help='name of the data file (default: test)')
elif args.model == "UNet":
    parser.add_argument('--filename', type=str, default="test", help='name of the data file (default: test)')
    parser.add_argument('--bilinear', type=str, default="True", help='if true, decoder is bilinear (default: "True")')

args = parser.parse_args()
# data handler
if args.load_data == "True":
    data_handler = Data_handler(file_name=args.filename, 
                                )
    train_loader = data_handler.train_loader
    validation_loader= data_handler.validation_loader
else:
    train_dataset = Dataset(num_samples_per_epoch = 500,h = args.image_size, w = args.image_size)
    validation_dataset = Dataset(num_samples_per_epoch=50,h = args.image_size, w = args.image_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, #num_workers=75
                                                )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size,# num_workers=75
                                                        )
# create autoenocder 
if args.model == "AE":
    if args.pretrained == "False":
        autoencoder = AE(base_channel_size=args.base_channel_size, height = args.image_size, width = args.image_size,lr=args.lr, layers=args.layers, bottleneck=args.bottleneck)
    else: 
        autoencoder = AE.load_from_checkpoint(args.pretrained)
elif args.model == "UNet":
    if args.pretrained == "False":
        autoencoder = UNet(base_channel_size=args.base_channel_size,height = args.image_size, width = args.image_size,lr=args.lr,bilinear = args.bilinear)
    else: 
        autoencoder = UNet.load_from_checkpoint(args.pretrained)
else: 
    print("ERROR: Model name invalid.")
    quit()
logger = CSVLogger("logs", name=args.model)


# train model
checkpoint_callback = ModelCheckpoint(monitor="val_loss")
trainer = pl.Trainer(max_epochs=args.epochs,logger=logger,callbacks=[checkpoint_callback,EarlyStopping(monitor="val_loss", mode="min",check_finite =True, patience=args.patience)])

trainer.fit(model=autoencoder, train_dataloaders=train_loader,val_dataloaders=validation_loader,)

# visualize model with one imgage
def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs[0].unsqueeze(0).to(model.device),input_imgs[2].unsqueeze(0).to(model.device))
    reconst_imgs = reconst_imgs[0].cpu()
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(6, 3))
    img = input_imgs[0].numpy()
    axes[0].imshow(img[0], cmap="Greys")
    axes[0].set_title("Original, theta = "+ str(input_imgs[2].squeeze(0).numpy()))

    img = input_imgs[1].numpy()
    axes[1].imshow(img[0], cmap="Greys")
    axes[1].set_title("Original")
    reconst_imgs = reconst_imgs.cpu()[0]
    
    axes[2].imshow(reconst_imgs[0].numpy(), cmap="Greys")
    axes[2].set_title("Reconstructed")
    plt.show()
    plt.savefig(str(args)+'.png')

# plot and save model
data_handler = Data_handler(file_name="noise_256_plot")
for data in data_handler.dataset:
    visualize_reconstructions(autoencoder,data)
    break