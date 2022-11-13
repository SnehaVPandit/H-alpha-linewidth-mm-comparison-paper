
import warnings
warnings.filterwarnings('ignore')

###NUMPY
import numpy as np
import numpy.ma as ma

###Usual matplotlib tools
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.patches import Rectangle
import matplotlib as mpl
from PIL import Image

###SCIPY
import scipy
from scipy import ndimage
from scipy.ndimage import convolve
from scipy import signal

###AstroPy tools for FITS, units, etc
import astropy.units as u
import astropy.constants as const
from astropy.io import fits

###Importing Sunpy and its colormaps. This allow us to use same SDO colormaps
import sunpy
#import sunpy.cm as cm
import sunpy.map

### for reading the Muilt3D output
from helita.vis import rh15d_vis
from helita.sim import multi3d

#MISCELANEOUS
import tqdm
import time
import glob
from numba import jit
import csv
import h5py
from os.path import splitext
import xarray as xr
     

with open(r"/mn/stornext/u3/snehap/rh/rh15d/run/output/H_alpha_linewidth_data.txt") as datFile:
    blue_wing=np.asarray([data.split(',')[2] for data in datFile])

blue_wing_ori=blue_wing[1:].astype(np.float)

blue_wing=blue_wing_ori.reshape((504,504))
with open(r"/mn/stornext/u3/snehap/rh/rh15d/run/output/H_alpha_linewidth_data.txt") as datFile:
    red_wing=np.asarray([data.split(',')[3] for data in datFile])

red_wing_ori=red_wing[1:].astype(np.float)

red_wing=red_wing_ori.reshape((504,504))


print(np.max(blue_wing),np.min(blue_wing),np.max(red_wing),np.min(red_wing))
    
plt.rcParams["figure.figsize"] = (13,10)

plt.imshow((red_wing-6564).T, origin='lower',cmap='gray',vmin=np.nanpercentile((red_wing-6564).flatten(),1),vmax=np.nanpercentile((red_wing-6564).flatten(),99))
cbar=plt.colorbar()
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('X (Mm)',fontsize=25)
plt.ylabel('Y (Mm)',fontsize=25)
#plt.title('H alpha red wing',fontsize=25)
cbar.set_label('H alpha blue wing ($\AA$ +6564$\AA$)', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Ha_blue_wing_map_orig_res.png')
plt.close()

plt.rcParams["figure.figsize"] = (13,10)
plt.tight_layout()
plt.imshow((blue_wing-6565).T, origin='lower',cmap='gray',vmin=np.nanpercentile((blue_wing-6565).flatten(),1),vmax=np.nanpercentile((blue_wing-6565).flatten(),99))
cbar=plt.colorbar()
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('X (Mm)',fontsize=25)
plt.ylabel('Y (Mm)',fontsize=25)
#plt.title('H alpha blue wing',fontsize=25)
cbar.set_label('H alpha red wing ($\AA$ +6565$\AA$)', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.savefig('Ha_red_wing_map_orig_res.png')
plt.close()
