import h5py
import numpy as np
import matplotlib.pyplot as plt
from helita.sim import rh15d
import astropy.units as u
import glob
from os.path import splitext
import astropy.constants as const
from helita.vis import rh15d_vis
import xarray as xr
from scipy.ndimage import convolve
import numpy.ma as ma
from scipy import signal
from astropy.io import fits
import radio_beam as rb
import seaborn as sns
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import csv
from matplotlib.patches import Rectangle
from PIL import Image
from IPython import embed

###Extra
import warnings
warnings.filterwarnings('ignore')

###NUMPY
import numpy as np
#import salat

###Dask 
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

###Usual matplotlib tools
import matplotlib
import matplotlib.pyplot as plt

###SCIPY
import scipy
from scipy import ndimage
from scipy.ndimage import convolve

###AstroPy tools for FITS, units, etc
import astropy.units as u
from astropy.io import fits

###Importing Sunpy and its colormaps. This allow us to use same SDO colormaps
import sunpy
#import sunpy.cm as cm
import sunpy.map

# ##IDL READER
# import idlsave

###BEAM CREATOR
import radio_beam as rb

### for reading the Muilt3D output
from helita.vis import rh15d_vis
from helita.sim import multi3d

#MISCELANEOUS
import tqdm
import time
import glob
from numba import jit
import csv

#reading the S index and IRT index files

with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    lambd=np.asarray([data.split(',')[0] for data in datFile])

lambd=np.asarray(lambd[1:]).astype(np.float)
    
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation_original_res=np.asarray([data.split(',')[1] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation_quiet_box_original_res=np.asarray([data.split(',')[2] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation_network_box_original_res=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation=np.asarray([data.split(',')[4] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation_quiet_box=np.asarray([data.split(',')[5] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_S_mm_Q_EN.txt") as datFile:
    correlation_network_box=np.asarray([data.split(',')[6] for data in datFile])
    

with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope_original_res=np.asarray([data.split(',')[1] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope_quiet_box_original_res=np.asarray([data.split(',')[2] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope_network_box_original_res=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope=np.asarray([data.split(',')[4] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope_quiet_box=np.asarray([data.split(',')[5] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_S_mm_Q_EN.txt") as datFile:
    slope_network_box=np.asarray([data.split(',')[6] for data in datFile])

with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT_original_res=np.asarray([data.split(',')[1] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT_quiet_box_original_res=np.asarray([data.split(',')[2] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT_network_box_original_res=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT=np.asarray([data.split(',')[4] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT_quiet_box=np.asarray([data.split(',')[5] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/correlations_CaII_IRT_mm_Q_EN.txt") as datFile:
    correlation_IRT_network_box=np.asarray([data.split(',')[6] for data in datFile])
    

with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT_original_res=np.asarray([data.split(',')[1] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT_quiet_box_original_res=np.asarray([data.split(',')[2] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT_network_box_original_res=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT=np.asarray([data.split(',')[4] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT_quiet_box=np.asarray([data.split(',')[5] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/slopes_intercepts_CaII_IRT_mm_Q_EN.txt") as datFile:
    slope_IRT_network_box=np.asarray([data.split(',')[6] for data in datFile])
# S index che slopes ani correlations

correlation=np.asarray(correlation[1:]).astype(np.float)
correlation_quiet_box=np.asarray(correlation_quiet_box[1:]).astype(np.float)
correlation_network_box=np.asarray(correlation_network_box[1:]).astype(np.float)
correlation_network_box_original_res=np.asarray(correlation_network_box_original_res[1:]).astype(np.float)
correlation_original_res=np.asarray(correlation_original_res[1:]).astype(np.float)
correlation_quiet_box_original_res=np.asarray(correlation_quiet_box_original_res[1:]).astype(np.float)

plt.rcParams["figure.figsize"] = (11,10)

plt.plot(lambd,correlation_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,correlation_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,correlation_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,correlation,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,correlation_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,correlation_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.0,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=1.6,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('s index mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_s_index_correlations_Q_EN_plot_.png')
plt.close()
print('s index correlations cha plot kela')

slope=np.asarray(slope[1:]).astype(np.float)
slope_quiet_box=np.asarray(slope_quiet_box[1:]).astype(np.float)
slope_network_box=np.asarray(slope_network_box[1:]).astype(np.float)
slope_network_box_original_res=np.asarray(slope_network_box_original_res[1:]).astype(np.float)
slope_original_res=np.asarray(slope_original_res[1:]).astype(np.float)
slope_quiet_box_original_res=np.asarray(slope_quiet_box_original_res[1:]).astype(np.float)

plt.rcParams["figure.figsize"] = (11,10)

plt.plot(lambd,slope_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,slope_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,slope_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,slope,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,slope_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,slope_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')
plt.axhline(y=0.0,linestyle='--',color='k')
plt.axvline(x=3.0,linestyle='--',color='k')
#plt.axvline(x=3.6,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
#seaborn.regplot(waves, correl, color='k', marker='+')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.title('s index mm slopes',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('mm_bands_s_index_slopes_Q_EN_plot.png')
plt.close()
#print('correlations cha plot kela')
print('s index slopes cha plot kela')



# IRT index che slopes ani correlations

correlation_IRT=np.asarray(correlation_IRT[1:]).astype(np.float)
correlation_IRT_quiet_box=np.asarray(correlation_IRT_quiet_box[1:]).astype(np.float)
correlation_IRT_network_box=np.asarray(correlation_IRT_network_box[1:]).astype(np.float)
correlation_IRT_network_box_original_res=np.asarray(correlation_IRT_network_box_original_res[1:]).astype(np.float)
correlation_IRT_original_res=np.asarray(correlation_IRT_original_res[1:]).astype(np.float)
correlation_IRT_quiet_box_original_res=np.asarray(correlation_IRT_quiet_box_original_res[1:]).astype(np.float)

plt.rcParams["figure.figsize"] = (11,10)

plt.plot(lambd,correlation_IRT_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,correlation_IRT_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,correlation_IRT_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,correlation_IRT,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,correlation_IRT_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,correlation_IRT_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.0,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=1.6,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('IRT index mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_IRT_index_correlations_Q_EN_plot.png')
plt.close()
print('IRT index correlations cha plot kela')

slope_IRT=np.asarray(slope_IRT[1:]).astype(np.float)
slope_IRT_quiet_box=np.asarray(slope_IRT_quiet_box[1:]).astype(np.float)
slope_IRT_network_box=np.asarray(slope_IRT_network_box[1:]).astype(np.float)
slope_IRT_network_box_original_res=np.asarray(slope_IRT_network_box_original_res[1:]).astype(np.float)
slope_IRT_original_res=np.asarray(slope_IRT_original_res[1:]).astype(np.float)
slope_IRT_quiet_box_original_res=np.asarray(slope_IRT_quiet_box_original_res[1:]).astype(np.float)

plt.rcParams["figure.figsize"] = (13,10)

plt.plot(lambd,slope_IRT_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,slope_IRT_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,slope_IRT_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,slope_IRT,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,slope_IRT_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,slope_IRT_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')
plt.axhline(y=0.0,linestyle='--',color='k')
plt.axvline(x=3.0,linestyle='--',color='k')
#plt.axvline(x=3.6,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
#seaborn.regplot(waves, correl, color='k', marker='+')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.title('IRT index mm slopes',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('mm_bands_IRT_index_slopes_Q_EN_plot.png')
plt.close()
print('IRT index slopes cha plot kela')

print('zala sagla.')
