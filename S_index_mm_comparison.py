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

def Cal_FWHM(wavelength_in_mm):
    FWHM=2*wavelength_in_mm/3
    return FWHM

def beam_kernel_calulator(bmaj_obs,bmin_obs,bpan_obs,ART_pxsz):
	"""
	Calculate the beam array using the observed beam to be used for convolving the ART data
	"""
	beam = rb.Beam(bmaj_obs*u.arcsec,bmin_obs*u.arcsec,bpan_obs*u.deg)
	beam_kernel = np.asarray(beam.as_kernel(pixscale=ART_pxsz*u.arcsec))
	return beam_kernel

def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='r',linewidth=2.0,label="Whole box")

def abline_QS(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='g',linewidth=2.0,label="Quiet Sun")

def abline_EN(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='b',linewidth=2.0,label="Network region")

x=[]
y=[]

s=504
for i in range(s):
    x.append(i)
    y.append(i)
print('x ani y arrays banavle')

list_mb = ["wavelength",'original resolution slope','Original Quiet slope','Original Network slope','ALMA res full box slope','ALMA res Quiet sun box slope','ALMA res Network region box slope','original resolution intercept','Original Quiet intercept','Original Network intercept','ALMA res full box intercept','ALMA res Quiet sun box intercept','ALMA res Network region box intercept']
file_slope = open("slopes_intercepts_CaII_S_mm_Q_EN.txt", "w")
writer_mb_S_index = csv.writer(file_slope)
writer_mb_S_index.writerow(list_mb)

list_corr=["wavelength",'original resolution','Original Quiet','Original Network','ALMA res full box','ALMA res Quiet sun box','ALMA res Network region box']
file = open("correlations_CaII_S_mm_Q_EN.txt", "w")
writer_corr_S_index = csv.writer(file)
writer_corr_S_index.writerow(list_corr)

list_mb = ["wavelength",'original resolution slope','Original Quiet slope','Original Network slope','ALMA res full box slope','ALMA res Quiet sun box slope','ALMA res Network region box slope','original resolution intercept','Original Quiet intercept','Original Network intercept','ALMA res full box intercept','ALMA res Quiet sun box intercept','ALMA res Network region box intercept']
file_slope = open("slopes_intercepts_CaII_IRT_mm_Q_EN.txt", "w")
writer_mb_IRT_index = csv.writer(file_slope)
writer_mb_IRT_index.writerow(list_mb)

list_corr=["wavelength",'original resolution','Original Quiet','Original Network','ALMA res full box','ALMA res Quiet sun box','ALMA res Network region box']
file = open("correlations_CaII_IRT_mm_Q_EN.txt", "w")
writer_corr_IRT_index = csv.writer(file)
writer_corr_IRT_index.writerow(list_corr)

correlation_original_res=[]
correlation_quiet_box_original_res=[]
correlation_network_box_original_res=[]
correlation=[]
correlation_quiet_box=[]
correlation_network_box=[]

slope_original_res=[]
slope_quiet_box_original_res=[]
slope_network_box_original_res=[]
slope=[]
slope_quiet_box=[]
slope_network_box=[]
lambd=[]

correlation_IRT_original_res=[]
correlation_IRT_quiet_box_original_res=[]
correlation_IRT_network_box_original_res=[]
correlation_IRT=[]
correlation_IRT_quiet_box=[]
correlation_IRT_network_box=[]

slope_IRT_original_res=[]
slope_IRT_quiet_box_original_res=[]
slope_IRT_network_box_original_res=[]
slope_IRT=[]
slope_IRT_quiet_box=[]
slope_IRT_network_box=[]

with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    x=np.asarray([data.split(',')[0] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    y=np.asarray([data.split(',')[1] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    R=np.asarray([data.split(',')[2] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    V=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    H=np.asarray([data.split(',')[4] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    K=np.asarray([data.split(',')[5] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    VR=np.asarray([data.split(',')[6] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    HK=np.asarray([data.split(',')[7] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/s_index/s_index_data.txt") as datFile:
    S=np.asarray([data.split(',')[8].strip() for data in datFile])

S = S[1:].astype(np.float)
S_index=S.reshape((504,504))
    
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    R_IRT=np.asarray([data.split(',')[2].strip() for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    V_IRT=np.asarray([data.split(',')[3].strip() for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    IRT_1=np.asarray([data.split(',')[4].strip() for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    IRT_2=np.asarray([data.split(',')[5].strip() for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    IRT_3=np.asarray([data.split(',')[6].strip() for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/rh/IRT_index/Ca_II_convolved_IRT_index_data.txt") as datFile:
    IRT=np.asarray([data.split(',')[7].strip() for data in datFile])

    
IRT = IRT[1:].astype(np.float)
IRT_index=IRT.reshape((504,504))

data = h5py.File("/mn/stornext/d18/RoCS/snehap/mm_data_from_BIFROST/bifrost_en024048_1000_int_sneha.h5",'r')
wave = data['Wavelength'][:] * u.angstrom
freq = wave.to(u.GHz,equivalencies=u.spectral())
wmm = wave.to(u.mm)
waves=wmm.value
for idx, iw in enumerate(wmm):
    print(idx," ->", iw)
scale_factor = (const.c.cgs**2 / (2*(freq.cgs**2)*const.k_B.cgs)).value
scale_factor

data_in_K = np.array(data["Stokes_I"][:,:,:,:]) * scale_factor[:]

for i in range(35):
    data_in_K[:,:,:,i][np.isnan(data_in_K[:,:,:,i])] = np.nanmean(data_in_K[:,:,:,i])


flist_mm = sorted(glob.glob('/mn/stornext/u3/snehap/alma_plots/ALMA_Molnar/Ha_linewidth_correct_scatter_plots/BIFROST_mm_Molnar_LOWRES_*.npy'))

dat789= xr.open_dataset('/mn/stornext/d18/RoCS/snehap/rh_first/rh15d/run_parallel_6/output_backup/output_ray.hdf5')

wave_789 = dat789.wavelength
indices_789 = np.arange(len(wave_789))

n_wave=len(wave_789)

ART_pxsz = 0.066

V_low, V_high=29,304
K_low, K_high=315,419
H_low, H_high=442,540
R_low, R_high=554,829

IRT_V_low,IRT_V_high=831,846
IRT_1_low,IRT_1_high=854,905
IRT_2_low,IRT_2_high=924,967
IRT_3_low,IRT_3_high=988,1036
IRT_R_low,IRT_R_high=1045,1062

triangle_K=np.zeros(len(wave_789))
K_range=K_high-K_low
num_K=np.int(K_range/2)
a1=np.linspace(0.0, 1.0, num=num_K)
for i in range(num_K):
    triangle_K[i+K_low]=a1[i]
b1=np.linspace(1.0, 0.0, num=num_K)
for i in range(num_K):
    triangle_K[i+K_low+num_K]=b1[i]

triangle_H=np.zeros(len(wave_789))
H_range=H_high-H_low
num_H=np.int(H_range/2)
a2=np.linspace(0.0, 0.5, num=num_H)
for i in range(0,num_H,1):
    triangle_H[i+H_low]=a2[i]
b2=np.linspace(0.5, 0.0, num=num_H)
for i in range(0,num_H,1):
    triangle_H[i+H_low+num_H]=b2[i]

s=504  #size of the box in pixels

for wave_index in range(1,35):
    print(wave_index)
    
    bmaj_obs,bmin_obs,bpan_obs = Cal_FWHM(waves[wave_index]),Cal_FWHM(waves[wave_index]),0.00
    beam_kernel = beam_kernel_calulator(bmaj_obs,bmin_obs,bpan_obs,ART_pxsz)

    print("Kernel banavla.")

    delayed_results = np.empty([s,s,n_wave],dtype=float)

    for l in range(n_wave):
        conv_results = convolve(dat789.intensity[:,:,l],beam_kernel)
        delayed_results[:,:,l]=conv_results[:,:]
        #print(j)

    print("Convolution karunn zala")
    
    np.save("./BIFROST_Ca_II_LOWRES_%s.npy" %np.around(waves[wave_index], decimals=2),np.array(delayed_results),allow_pickle=True)
    print("poorna Ca II chi npy file lihili")


    s_index_789_all=np.empty([s,s],dtype=float)
    H_789_all=np.empty([s,s],dtype=float)
    K_789_all=np.empty([s,s],dtype=float)
    R_789_all=np.empty([s,s],dtype=float)
    V_789_all=np.empty([s,s],dtype=float)
    V_R_all=np.empty([s,s],dtype=float)
    H_K_all=np.empty([s,s],dtype=float)
    
    IRT_789_all=np.empty([s,s],dtype=float)
    _1_IRT_789_all=np.empty([s,s],dtype=float)
    _2_IRT_789_all=np.empty([s,s],dtype=float)
    _3_IRT_789_all=np.empty([s,s],dtype=float)
    R_IRT_789_all=np.empty([s,s],dtype=float)
    V_IRT_789_all=np.empty([s,s],dtype=float)
    
    for i in range(s):
        for j in range(s):
            k=delayed_results[i,j] * triangle_K
            h=delayed_results[i,j] * triangle_H
    
            inti_V=0.00
            for l in range(29,304):
                inti_V=inti_V+delayed_results[i,j,l]
            inti_V=inti_V/(304-29)
    
            inti_K=0.00
            for l in range(315,419):
                inti_K=inti_K+k[l]
            inti_K=inti_K/(419-315)
    
            inti_H=0.00
            for l in range(442,540):
                inti_H=inti_H+h[l]
            inti_H=inti_H/(540-442)
    
            inti_R=0.00
            for l in range(554,829):
                inti_R=inti_R+delayed_results[i,j,l]
            inti_R=inti_R/(829-554)
            
            V_789_all[i][j]=(inti_V)
            K_789_all[i][j]=(inti_K)
            H_789_all[i][j]=(inti_H)
            R_789_all[i][j]=(inti_R)
            H_K_all[i][j]=(inti_H+inti_K)
            V_R_all[i][j]=(inti_V+inti_R)
            s_index_789_all[i][j]=(2.4*8.0*(inti_H+inti_K)/(inti_V+inti_R))
            s_index_789_all[np.isnan(s_index_789_all)] = 0.0
            
            V_IRT_789=(np.nanmean(delayed_results[i,j,IRT_V_low:IRT_V_high]))*0.5
            _1_IRT_789=(np.nanmean(delayed_results[i,j,IRT_1_low:IRT_1_high]))*0.2
            _2_IRT_789=(np.nanmean(delayed_results[i,j,IRT_2_low:IRT_2_high]))*0.2
            _3_IRT_789=(np.nanmean(delayed_results[i,j,IRT_3_low:IRT_3_high]))*0.2
            R_IRT_789=(np.nanmean(delayed_results[i,j,IRT_R_low:IRT_R_high]))*0.5
            V_IRT_789_all[i][j]=(V_IRT_789)
            _1_IRT_789_all[i][j]=(_1_IRT_789)
            _2_IRT_789_all[i][j]=(_2_IRT_789)
            _3_IRT_789_all[i][j]=(_3_IRT_789)
            R_IRT_789_all[i][j]=(R_IRT_789)
            IRT_789_all[i][j]=((_1_IRT_789+_2_IRT_789+_3_IRT_789)/(V_IRT_789+R_IRT_789))
            IRT_789_all[np.isnan(IRT_789_all)] = 0.0
    
    print(np.shape(IRT_789_all))
    print(np.shape(s_index_789_all))
    
    np.save("./s_index_LOWRES_%s.npy" %np.around(waves[wave_index], decimals=2),np.array(s_index_789_all),allow_pickle=True)
    print("s index chi npy file lihili")
    
    np.save("./IRT_index_LOWRES_%s.npy" %np.around(waves[wave_index], decimals=2),np.array(IRT_789_all),allow_pickle=True)
    print("IRT index chi npy file lihili")
            
    print("s ani IRT indices calculate karun zale. Ani lihun sudha zale.")
    
    plt.rcParams["figure.figsize"] = (13,10)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    plt.imshow(s_index_789_all, origin='lower',cmap='gist_yarg', vmax=1.0)
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('s index',fontsize=25)
    #cbar=plt.colorbar()
    cbar.set_label('Calculated s index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    #plt.show()
    plt.savefig('s_index_map_truncated_for_res_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()

    plt.rcParams["figure.figsize"] = (13,10)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    plt.imshow(s_index_789_all, origin='lower',cmap='gist_yarg')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('s index',fontsize=25)
    #cbar=plt.colorbar()
    cbar.set_label('Calculated s index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    #plt.show()
    plt.savefig('s_index_map_for_res_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()

    plt.rcParams["figure.figsize"] = (13,10)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    plt.imshow(IRT_789_all, origin='lower',cmap='gist_yarg', vmax=1.0)
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('IRT index',fontsize=25)
    #cbar=plt.colorbar()
    cbar.set_label('Calculated IRT index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    #plt.show()
    plt.savefig('IRT_index_map_truncated_for_res_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()

    plt.rcParams["figure.figsize"] = (13,10)
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    plt.imshow(IRT_789_all, origin='lower',cmap='gist_yarg')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('IRT index',fontsize=25)
    #cbar=plt.colorbar()
    cbar.set_label('Calculated IRT index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    #plt.show()
    plt.savefig('IRT_index_map_for_res_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()
    
    print("s ani IRT indices che plots kele")
    
    linew=S_index
    
    '''
    #for original resolution data
    '''
    correlation_original_resolution=np.corrcoef(np.ndarray.flatten(linew),np.ndarray.flatten(np.asarray(data_in_K[:,:,:,wave_index].T)))
    correlation_original_res.append(np.corrcoef(np.ndarray.flatten(linew),np.ndarray.flatten(np.asarray(data_in_K[:,:,:,wave_index].T)))[0][1])
    correlation_quiet_ori_res=np.corrcoef(np.ndarray.flatten(linew[0:200,300:500]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[0:200,300:500])))
    correlation_quiet_box_original_res.append(correlation_quiet_ori_res[0][1])
    correlation_network_ori_res=np.corrcoef(np.ndarray.flatten(linew[140:340,170:370]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[140:340,170:370])))
    correlation_network_box_original_res.append(correlation_network_ori_res[0][1])

    ART_conv=np.asarray(np.load(flist_mm[wave_index-1]))
    linew_Lowres=np.asarray(s_index_789_all)
    '''
    #For data convolved with gaussian kernel
    '''    
    correlation_calc=np.corrcoef(np.ndarray.flatten(linew_Lowres),np.ndarray.flatten(ART_conv))
    correlation.append(correlation_calc[0][1])
    correlation_quiet=np.corrcoef(np.ndarray.flatten(linew_Lowres[0:200,300:500]),np.ndarray.flatten(ART_conv[0:200,300:500]))
    correlation_quiet_box.append(correlation_quiet[0][1])
    correlation_network_b=np.corrcoef(np.ndarray.flatten(linew_Lowres[140:340,170:370]),np.ndarray.flatten(ART_conv[140:340,170:370]))
    correlation_network_box.append(correlation_network_b[0][1])
    lambd.append(waves[wave_index])
    
    #embed()
    
    plt.rcParams['figure.figsize'] = [37,20]

    plt.subplot(2,3,1)
    plt.imshow(linew.T,origin='lower',vmin=np.nanpercentile((np.ndarray.flatten(linew)),1),vmax=np.nanpercentile((np.ndarray.flatten(linew)),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('s index',fontsize=25)
    cbar.set_label('linewidth (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,2)
    plt.imshow(data_in_K[0,:,:,wave_index]/1000,origin='lower',cmap='gray',vmin=np.nanpercentile(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000),1),vmax=np.nanpercentile((np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000)),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    plt.title('ART output %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,3)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000),np.ndarray.flatten(np.asarray(linew)),bins=300,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m, b = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T/1000)),np.ndarray.flatten(np.asarray(linew)) , 1)
    slope_original_res.append(m*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T[0:200,300:500]/1000),np.ndarray.flatten(linew[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_QS, b_QS = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[0:200,300:500]/1000)),np.ndarray.flatten(linew[0:200,300:500]) , 1)
    
    slope_quiet_box_original_res.append(m_QS*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T[160:360,140:340]/1000),np.ndarray.flatten(linew[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_EN, b_EN = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[160:360,140:340]/1000)),np.ndarray.flatten(linew[160:360,140:340]) , 1)
    slope_network_box_original_res.append(m_EN*1000)
    abline_EN(m_EN,b_EN)
    abline_QS(m_QS,b_QS)
    abline(m,b)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    plt.title('s index vs ALMA %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    #plt.xlim(4.7,11.7)
    #plt.ylim(0.85,1.35)
    #plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('s index',fontsize=25)
    
    plt.subplot(2,3,4)
    plt.imshow(linew_Lowres.T,origin='lower',vmin=np.nanpercentile((linew_Lowres),1),vmax=np.nanpercentile((linew_Lowres),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    #plt.title('H alpha linewidth (Angstrom)',fontsize=25)
    cbar.set_label('s index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,5)
    plt.imshow(ART_conv.T,origin='lower',cmap='gray',vmin=np.nanpercentile(np.ndarray.flatten(ART_conv),1),vmax=np.nanpercentile(np.ndarray.flatten(ART_conv),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    #plt.title('ART output %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,6)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(ART_conv),np.ndarray.flatten(linew_Lowres),bins=100,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m_LR, b_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv)),np.ndarray.flatten(linew_Lowres) , 1)
    slope.append(m_LR)
    sns.kdeplot(np.ndarray.flatten(ART_conv[0:200,300:500]),np.ndarray.flatten(linew_Lowres[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_QS_LR, b_QS_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[0:200,300:500])),np.ndarray.flatten(linew_Lowres[0:200,300:500]) , 1)
    slope_quiet_box.append(m_QS_LR)
    sns.kdeplot(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_EN_LR, b_EN_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]) , 1)
    slope_network_box.append(m_EN_LR)
    abline(m_LR,b_LR)
    abline_QS(m_QS_LR,b_QS_LR)
    abline_EN(m_EN_LR,b_EN_LR)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    #plt.title('Network region: Ha linewidth vs ALMA %s' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('s index',fontsize=25)
    plt.legend(fontsize=25)
    #plt.xlim(4.7,11.7)
    #plt.ylim(0.85,1.35)
    plt.tight_layout()
    #plt.show()
    plt.savefig('S_index_mm_scatter_plot_Both_res_6_plot_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()
    
    print('S index cha 6 scatter plot line saha plot kelay.')

    print([waves[wave_index],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])
    writer_corr_S_index.writerow([waves[wave_index],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])

    print([waves[wave_index],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    writer_mb_S_index.writerow([waves[wave_index],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    print('s index che slopes ani intercepts ani correlations lihile')

    linew=IRT_index
    
    '''
    #for original resolution data
    '''
    correlation_IRT_original_resolution=np.corrcoef(np.ndarray.flatten(linew),np.ndarray.flatten(np.asarray(data_in_K[:,:,:,wave_index].T)))
    correlation_IRT_original_res.append(np.corrcoef(np.ndarray.flatten(linew),np.ndarray.flatten(np.asarray(data_in_K[:,:,:,wave_index].T)))[0][1])
    correlation_IRT_quiet_ori_res=np.corrcoef(np.ndarray.flatten(linew[0:200,300:500]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[0:200,300:500])))
    correlation_IRT_quiet_box_original_res.append(correlation_IRT_quiet_ori_res[0][1])
    correlation_IRT_network_ori_res=np.corrcoef(np.ndarray.flatten(linew[140:340,170:370]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[140:340,170:370])))
    correlation_IRT_network_box_original_res.append(correlation_IRT_network_ori_res[0][1])

    ART_conv=np.asarray(np.load(flist_mm[wave_index-1]))
    linew_Lowres=np.asarray(IRT_789_all)
    '''
    #For data convolved with gaussian kernel
    '''    
    correlation_IRT_calc=np.corrcoef(np.ndarray.flatten(linew_Lowres),np.ndarray.flatten(ART_conv))
    correlation_IRT.append(correlation_calc[0][1])
    correlation_IRT_quiet=np.corrcoef(np.ndarray.flatten(linew_Lowres[0:200,300:500]),np.ndarray.flatten(ART_conv[0:200,300:500]))
    correlation_IRT_quiet_box.append(correlation_IRT_quiet[0][1])
    correlation_IRT_network_b=np.corrcoef(np.ndarray.flatten(linew_Lowres[140:340,170:370]),np.ndarray.flatten(ART_conv[140:340,170:370]))
    correlation_IRT_network_box.append(correlation_IRT_network_b[0][1])
    
    plt.rcParams['figure.figsize'] = [37,20]

    plt.subplot(2,3,1)
    plt.imshow(linew.T,origin='lower',vmin=np.nanpercentile((np.ndarray.flatten(linew)),1),vmax=np.nanpercentile((np.ndarray.flatten(linew)),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('IRT index',fontsize=25)
    cbar.set_label('linewidth (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,2)
    plt.imshow(data_in_K[0,:,:,wave_index]/1000,origin='lower',cmap='gray',vmin=np.nanpercentile(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000),1),vmax=np.nanpercentile((np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000)),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    plt.title('ART output %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,3)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T/1000),np.ndarray.flatten(np.asarray(linew)),bins=300,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m_IRT, b_IRT = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T/1000)),np.ndarray.flatten(np.asarray(linew)) , 1)
    slope_IRT_original_res.append(m*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T[0:200,300:500]/1000),np.ndarray.flatten(linew[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_IRT_QS, b_IRT_QS = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[0:200,300:500]/1000)),np.ndarray.flatten(linew[0:200,300:500]) , 1)
    
    slope_IRT_quiet_box_original_res.append(m_QS*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,wave_index].T[160:360,140:340]/1000),np.ndarray.flatten(linew[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_IRT_EN, b_IRT_EN = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,wave_index].T[160:360,140:340]/1000)),np.ndarray.flatten(linew[160:360,140:340]) , 1)
    slope_IRT_network_box_original_res.append(m_EN*1000)
    abline_EN(m_IRT_EN,b_IRT_EN)
    abline_QS(m_IRT_QS,b_IRT_QS)
    abline(m_IRT,b_IRT)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    plt.title('IRT index vs ALMA %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    #plt.xlim(4.7,11.7)
    #plt.ylim(0.85,1.35)
    #plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('IRT index',fontsize=25)
    
    plt.subplot(2,3,4)
    plt.imshow(linew_Lowres.T,origin='lower',vmin=np.nanpercentile((linew_Lowres),1),vmax=np.nanpercentile((linew_Lowres),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    #plt.title('H alpha linewidth (Angstrom)',fontsize=25)
    cbar.set_label('IRT index', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,5)
    plt.imshow(ART_conv.T,origin='lower',cmap='gray',vmin=np.nanpercentile((np.ndarray.flatten(ART_conv)),1),vmax=np.nanpercentile((np.ndarray.flatten(ART_conv)),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    #plt.title('ART output %s mm' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,6)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(ART_conv),np.ndarray.flatten(linew_Lowres),bins=100,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m_IRT_LR, b_IRT_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv)),np.ndarray.flatten(linew_Lowres) , 1)
    slope_IRT.append(m_LR)
    sns.kdeplot(np.ndarray.flatten(ART_conv[0:200,300:500]),np.ndarray.flatten(linew_Lowres[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_IRT_QS_LR, b_IRT_QS_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[0:200,300:500])),np.ndarray.flatten(linew_Lowres[0:200,300:500]) , 1)
    slope_IRT_quiet_box.append(m_QS_LR)
    sns.kdeplot(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_IRT_EN_LR, b_IRT_EN_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]) , 1)
    slope_IRT_network_box.append(m_EN_LR)
    abline(m_IRT_LR,b_IRT_LR)
    abline_QS(m_IRT_QS_LR,b_IRT_QS_LR)
    abline_EN(m_IRT_EN_LR,b_IRT_EN_LR)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    #plt.title('Network region: Ha linewidth vs ALMA %s' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('IRT index',fontsize=25)
    plt.legend(fontsize=25)
    #plt.xlim(4.7,11.7)
    #plt.ylim(0.85,1.35)
    plt.tight_layout()
    #plt.show()
    plt.savefig('IRT_index_mm_scatter_plot_Both_res_6_plot_%s.png' %np.around(waves[wave_index], decimals=2))
    plt.close()
    
    print('IRT index cha 6 scatter plot line saha plot kelay.')

    print([waves[wave_index],correlation_IRT_original_resolution[0][1],correlation_IRT_quiet_ori_res[0][1],correlation_IRT_network_ori_res[0][1],correlation_IRT_calc[0][1],correlation_IRT_quiet[0][1],correlation_IRT_network_b[0][1]])
    writer_corr_IRT_index.writerow([waves[wave_index],correlation_IRT_original_resolution[0][1],correlation_IRT_quiet_ori_res[0][1],correlation_IRT_network_ori_res[0][1],correlation_IRT_calc[0][1],correlation_IRT_quiet[0][1],correlation_IRT_network_b[0][1]])

    print([waves[wave_index],m_IRT,m_IRT_QS,m_IRT_EN,m_IRT_LR,m_IRT_QS_LR,m_IRT_EN_LR,b_IRT,b_IRT_QS,b_IRT_EN,b_IRT_LR,b_IRT_QS_LR,b_IRT_EN_LR])
    writer_mb_IRT_index.writerow([waves[wave_index],m_IRT,m_IRT_QS,m_IRT_EN,m_IRT_LR,m_IRT_QS_LR,m_IRT_EN_LR,b_IRT,b_IRT_QS,b_IRT_EN,b_IRT_LR,b_IRT_QS_LR,b_IRT_EN_LR])
    print('IRT index che slopes ani intercepts ani correlations lihile')
    
    
#S index che slopes ani correlations
              
correlation=np.asarray(correlation).astype(np.float)
correlation_quiet_box=np.asarray(correlation_quiet_box).astype(np.float)
correlation_network_box=np.asarray(correlation_network_box).astype(np.float)
correlation_network_box_original_res=np.asarray(correlation_network_box_original_res).astype(np.float)
correlation_original_res=np.asarray(correlation_original_res).astype(np.float)
correlation_quiet_box_original_res=np.asarray(correlation_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

plt.plot(lambd,correlation_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,correlation_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,correlation_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,correlation,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,correlation_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,correlation_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.84,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=2.2,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('s index mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_s_index_correlations_Q_EN_plot_.png')
plt.close()
print('s index correlations cha plot kela')

slope=np.asarray(slope).astype(np.float)
slope_quiet_box=np.asarray(slope_quiet_box).astype(np.float)
slope_network_box=np.asarray(slope_network_box).astype(np.float)
slope_network_box_original_res=np.asarray(slope_network_box_original_res).astype(np.float)
slope_original_res=np.asarray(slope_original_res).astype(np.float)
slope_quiet_box_original_res=np.asarray(slope_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

plt.plot(lambd,slope_original_res,marker='.',markersize=10,linewidth=2,label='Original resolution')
plt.plot(lambd,slope,marker='.',markersize=10,linewidth=2,label='ALMA resolution')
plt.plot(lambd,slope_quiet_box_original_res,marker='.',markersize=10,linewidth=2,label='Quiet sun Original')
plt.plot(lambd,slope_quiet_box,marker='.',markersize=10,linewidth=2,label='Quiet sun ALMA res')
plt.plot(lambd,slope_network_box_original_res,marker='.',markersize=10,linewidth=2,label='Network region original')
plt.plot(lambd,slope_network_box,marker='.',markersize=10,linewidth=2,label='Network region ALMA res')
plt.axhline(y=0.0612,linestyle='--',color='k')
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
#plt.close()
#print('correlations cha plot kela')
print('s index slopes cha plot kela')



#IRT index che slopes ani correlations
    
correlation_IRT=np.asarray(correlation_IRT).astype(np.float)
correlation_IRT_quiet_box=np.asarray(correlation_IRT_quiet_box).astype(np.float)
correlation_IRT_network_box=np.asarray(correlation_IRT_network_box).astype(np.float)
correlation_IRT_network_box_original_res=np.asarray(correlation_IRT_network_box_original_res).astype(np.float)
correlation_IRT_original_res=np.asarray(correlation_IRT_original_res).astype(np.float)
correlation_IRT_quiet_box_original_res=np.asarray(correlation_IRT_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

plt.plot(lambd,correlation_IRT_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd,correlation_IRT_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd,correlation_IRT_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,correlation_IRT,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,correlation_IRT_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,correlation_IRT_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.84,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=2.2,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('IRT index mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_IRT_index_correlations_Q_EN_plot.png')
plt.close()
print('IRT index correlations cha plot kela')

slope_IRT=np.asarray(slope_IRT).astype(np.float)
slope_IRT_quiet_box=np.asarray(slope_IRT_quiet_box).astype(np.float)
slope_IRT_network_box=np.asarray(slope_IRT_network_box).astype(np.float)
slope_IRT_network_box_original_res=np.asarray(slope_IRT_network_box_original_res).astype(np.float)
slope_IRT_original_res=np.asarray(slope_IRT_original_res).astype(np.float)
slope_IRT_quiet_box_original_res=np.asarray(slope_IRT_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

plt.plot(lambd,slope_IRT_original_res,marker='.',markersize=10,linewidth=2,label='Original resolution')
plt.plot(lambd,slope_IRT,marker='.',markersize=10,linewidth=2,label='ALMA resolution')
plt.plot(lambd,slope_IRT_quiet_box_original_res,marker='.',markersize=10,linewidth=2,label='Quiet sun Original')
plt.plot(lambd,slope_IRT_quiet_box,marker='.',markersize=10,linewidth=2,label='Quiet sun ALMA res')
plt.plot(lambd,slope_IRT_network_box_original_res,marker='.',markersize=10,linewidth=2,label='Network region original')
plt.plot(lambd,slope_IRT_network_box,marker='.',markersize=10,linewidth=2,label='Network region ALMA res')
plt.axhline(y=0.0612,linestyle='--',color='k')
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
