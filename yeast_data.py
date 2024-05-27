import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.draw import ellipse
from random import randint, choice
from typing import List
import skimage as ski


class DataLoaderYeast():
    def __init__(self):
        self.data = None

    def generate_yeast_phantom(self, w: int=128, h: int=128, base_intensity: float=1) -> np.ndarray:
        """
        Generate a yeast cell phantom image.
        
        Args:
            w (int): Width of the image.
            h (int): Height of the image.
            base_intensity (float): Intensity value for the cell membrane and vacuole.
            
        Returns:
            np.ndarray: Yeast cell phantom image.
        """
        
        proj = np.zeros((w, h), dtype=float)
        
        # generate cell membrane
        # We allow 10% of w,h as offset
        # Size is 20-35% of each dimension
        # r,c are coordinates of center
        # rads are radius of the semi-axes
        r = int(w*0.5 - randint(int(-w*0.1), int(w*0.1)))
        c = int(h*0.5 - randint(int(-h*0.1), int(h*0.1)))
        r_rad = randint(int(w*0.25), int(w*0.34))
        c_rad = randint(int(h*0.25), int(h*0.34))
        border_size = int(w*0.02)
        
        # We generate the border, and overwrite with the inner
        membrane_border = ellipse(r,c, r_rad, c_rad, shape=proj.shape)
        membrane_inner = ellipse(r,c, r_rad-border_size, c_rad-border_size, shape=proj.shape)

        # Draw membrane
        proj[membrane_border] = base_intensity * 2
        proj[membrane_inner] = base_intensity
        
        # We can generate the same ellipse with smaller radii to find valid vacuole positions.
        # We generate the vac based on membrane size.
        vac_rad = randint(int(min(r_rad, c_rad)*0.4), int(min(r_rad, c_rad)*0.80) - border_size*2)
        valid_coords = ellipse(r,c, r_rad-border_size*2 - vac_rad, c_rad-border_size*2 - vac_rad, shape=proj.shape)
        
        r_vac = choice(valid_coords[0])
        c_vac = choice(valid_coords[1])
        vac = ellipse(r_vac, c_vac, vac_rad, vac_rad, shape=proj.shape)
        
        # Draw Vac
        proj[vac] = base_intensity * 0.5

        # Now we add some random number of lipid droplets
        lip_rad = int(w * 0.03)
        valid_coords = ellipse(r,c, r_rad - border_size*3 - lip_rad, c_rad - border_size*3 - lip_rad, shape=proj.shape)
        
        for i in range(randint(2,8)):
            r = choice(valid_coords[0])
            c = choice(valid_coords[1])
            lip = ellipse(r, c, lip_rad, lip_rad, shape=proj.shape)
            proj[lip] = base_intensity * 2
        return proj
    
    def generate_reconstructions(self,img: np.ndarray, angle: int, noise: float = 0.0,sigma: float = 0) -> List[np.ndarray]:
        """
        Generate a list of reconstructions from sinograms for different projection angles.
        
        Args:
            img (np.ndarray): Input image to generate sinograms and reconstructions from.
            angles (List[int]): List of projection angles for sinogram generation.
            
        Returns:
            List[np.ndarray]: List of reconstructed images corresponding to each angle.
        """
        recons = []
        # Generate projection angles
        theta = np.linspace(-angle // 2, angle // 2, max(img.shape))
        # Generate sinogram
        sinogram = radon(img, theta=theta, circle=False)
        sinogram = sinogram+ np.random.randn(*sinogram.shape) * noise
        sinogram = ski.filters.gaussian(sinogram, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
        # Perform back-projection to reconstruct the image
        recon = iradon(sinogram, theta=theta, circle=False)
        
        recons.append([recon,img,angle])
        return recons

    def generate_reconstructions_noise_table(self,img: np.ndarray) -> List[np.ndarray]:
        """
        Generate a list of reconstructions from sinograms for different projection angles.
        
        Args:
            img (np.ndarray): Input image to generate sinograms and reconstructions from.
            angles (List[int]): List of projection angles for sinogram generation.
            
        Returns:
            List[np.ndarray]: List of reconstructed images corresponding to each angle.
        """
        recons = []
        angle = 120
        # Generate projection angles
        theta = np.linspace(-angle // 2, angle // 2, max(img.shape))
        # Generate sinogram
        noise = 0
        for i in range(10):
            noise = i+noise
            for j in range(5):
                sigma = j
                sinogram = radon(img, theta=theta, circle=False)
                sinogram = sinogram+ np.random.randn(*sinogram.shape) * noise
                sinogram = ski.filters.gaussian(sinogram, sigma=(sigma, sigma), truncate=3.5, channel_axis=-1)
                # Perform back-projection to reconstruct the image
                recon = iradon(sinogram, theta=theta, circle=False)
                recons.append([recon,img,angle,noise,sigma])
        return recons

    def plot_example_data(self, w: int = 128, h: int = 128):
        fig, axes = plt.subplots(5,5, figsize=(15,15))
        for i in range(5):
            for j in range(5):
                axes[i][j].imshow(self.generate_yeast_phantom(w=w, h=h), cmap="Greys")
    
    def plot_reconstructions_example(self, w: int = 128, h: int = 128):
        # Show a sample range
        yeast = self.generate_yeast_phantom(w=w, h=h)
        thetas = [180,160,140,120,100,80]
        recons = self.generate_reconstructions(yeast, thetas)
        fig, axes = plt.subplots(2,3, figsize=(15,10))
        for i in range(6):
            axes[i//3][i%3].imshow(recons[i][0], cmap="Greys")
            axes[i//3][i%3].set_title(f"FBP θ={recons[i][1]}")

    def generate_data(self, w: int=128, h: int=128, base_intensity: float=1, angles: list[list] = [[180,2],[160,2],[140,1]], repeat: int = 5, noise: float = 0.0,sigma: float = 1) -> np.ndarray:
        data = []
        unrolled_angles = [num for num, count in angles for _ in range(count)]
        for _ in range(repeat):
            for angle in unrolled_angles:
                yeast = self.generate_yeast_phantom(w=w, h=h, base_intensity=base_intensity)
                recons = self.generate_reconstructions(yeast, angle, noise, sigma)
                data.extend(recons)    
        self.data = data
    
    def generate_data_noise_table(self, w: int=128, h: int=128, base_intensity: float=1) -> np.ndarray:
        data = []
        yeast = self.generate_yeast_phantom(w=w, h=h, base_intensity=base_intensity)
        recons = self.generate_reconstructions_noise_table(yeast)
        data.extend(recons)  
        self.data = data

    def generate_and_save_data(self, file_name:str = "reconstack",w: int=128, h: int=128, base_intensity: float=1, angles: list[list] = [[180,2],[160,2],[140,1]], repeat: int = 5, noise: float = 0.0, sigma: int = 1) -> np.ndarray:
        """
        Generate a list of reconstructions of given number of yeast cells from sinograms for different projection angles.
        
        Args:
            file_name (str): 
            w (int): Width of the image.
            h (int): Height of the image.
            base_intensity (float): Intensity value for the cell membrane and vacuole.
            min_angle (int): Start value for for theta 
            max_angle (int): End value for theta
            angles (list[list]): Reconstruction angles and number of repetetions as [[angle,repetetion],[angle,repetetion],...]
            repeat(int): number of itterations for angles
            noice(float): random noice added for each pixel

        Returns:
            List[np.ndarray]: List of reconstructed images corresponding to each angle.
        """
        data = []
        unrolled_angles = [num for num, count in angles for _ in range(count)]
        for _ in range(repeat):
            for angle in unrolled_angles:
                yeast = self.generate_yeast_phantom(w=w, h=h, base_intensity=base_intensity)
                recons = self.generate_reconstructions(yeast, angle,noise,sigma)
                data.extend(recons)    
        file_name = "data/"+file_name + ".npy"
        np.save(file_name, np.array(data, dtype=object), allow_pickle=True)
        self.data = data
    
    def load_data(self, file_name: str = "reconstack") -> np.ndarray:
        """
        Generate a list of reconstructions from sinograms for different projection angles.
        
        Args:
            file_name (str): numpy file with data created from generate_and_save_data().

        Returns:
            List[np.ndarray]: List of reconstructed images corresponding to each angle.
        """
        self.data = np.load("data/"+file_name+".npy", allow_pickle=True)
