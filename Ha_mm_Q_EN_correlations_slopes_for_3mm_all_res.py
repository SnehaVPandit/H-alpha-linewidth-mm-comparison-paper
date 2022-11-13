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

x=[]
y=[]

s=504
for i in range(s):
    x.append(i)
    y.append(i)
print('x ani y arrays banavle')

multi3d.Multi3dOut(
    inputfile='/mn/stornext/d18/RoCS/snehap/multi3d_Halpha/output/multi3d.input',
    directory='/mn/stornext/d18/RoCS/snehap/multi3d_Halpha',
    printinfo=True,
)

data = multi3d.Multi3dOut(directory='/mn/stornext/d18/RoCS/snehap/multi3d_Halpha')
data.readall()

data.set_transition(3, 2)
emergent_intensity = data.readvar('ie')
#source_function = data.readvar('snu')
tau1_height = data.readvar('zt1')

wave=[]
for i in range(101):
    wave.append(np.float(data.d.l[i]/u.angstrom))

#avg_ART_data  = ha_data
ART_pxsz = 0.066

data = h5py.File("/mn/stornext/d18/RoCS/snehap/mm_data_from_BIFROST/bifrost_en024048_1000_int_sneha.h5",'r')
waves = data['Wavelength'][:] * u.angstrom
freq = waves.to(u.GHz,equivalencies=u.spectral())
wmm = waves.to(u.mm)
for idx, iw in enumerate(wmm):
    print(idx," ->", iw)
scale_factor = (const.c.cgs**2 / (2*(freq.cgs**2)*const.k_B.cgs)).value
scale_factor

data_in_K = np.array(data["Stokes_I"][:,:,:,:]) * scale_factor[:]

for i in range(35):
    data_in_K[:,:,:,i][np.isnan(data_in_K[:,:,:,i])] = np.nanmean(data_in_K[:,:,:,i])

with open(r"/mn/stornext/u3/snehap/rh/rh15d/run/output/H_alpha_linewidth_data.txt") as datFile:
    Linewidth=np.asarray([data.split(',')[4] for data in datFile])
'''    
with open(r"/mn/stornext/d18/RoCS/snehap/H_alpha_index_data.txt") as datFile:
    Ha=np.asarray([data.split(',')[3] for data in datFile])
with open(r"/mn/stornext/d18/RoCS/snehap/H_alpha_index_data.txt") as datFile:
    index=np.asarray([data.split(',')[5] for data in datFile])
    
indices=index[1:].astype(np.float)

Halph=Ha[1:].astype(np.float)
'''
linewidths=Linewidth[1:].astype(np.float)

slopes=[]
lambd=[]

linew=linewidths.reshape((504,504))
max_values=np.where(linew>200)
for i in range(len(max_values[0])):
    linew[max_values[0][i],max_values[1][i]]=0.0
linewidths=np.ndarray.flatten(linew)

waves=wmm.value
print(waves)

'''list_2 = ["wavelength",'original resolution slope','Original Quiet slope','Original Network slope','ALMA res full box slope','ALMA res Quiet sun box slope','ALMA res Network region box slope','original resolution intercept','Original Quiet intercept','Original Network intercept','ALMA res full box intercept','ALMA res Quiet sun box intercept','ALMA res Network region box intercept']
file_slope = open("slopes_intercepts_Ha_mm_Q_EN_22mm_all_res.txt", "w")
writer_mb = csv.writer(file_slope)
writer_mb.writerow(list_2)


list_1 = ["wavelength",'original resolution correlation','QS original resolution','EN original resolution','Degraded resolution','QS degraded resolution','EN degraded resolution']
file = open("correlations_Ha_mm_Q_EN_22mm_all_res.txt", "w")
writer_corr = csv.writer(file)
writer_corr.writerow(list_1)'''

flist_Ha = sorted(glob.glob('/mn/stornext/u3/snehap/alma_plots/ALMA_Molnar/Ha_linewidth_correct_scatter_plots/BIFROST_H_alpha_linewidths_Molnar_LOWRES_*.npy'))

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
for i in range(1,35):
    
    bmaj_obs,bmin_obs,bpan_obs = Cal_FWHM(waves[i]),Cal_FWHM(waves[i]),0.00
    beam_kernel = beam_kernel_calulator(bmaj_obs,bmin_obs,bpan_obs,ART_pxsz)

    print("Kernel banavla.")    
    ART_conv=convolve((data_in_K[0,:,:,18].T)/1000,beam_kernel)
    print('mm data convolve kela')
    
    linew_Lowres=np.load(flist_Ha[i-1])
    
    '''
    for original resolution data
    '''
    correlation_original_resolution=np.corrcoef(linewidths,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,18].T)))
    correlation_original_res.append(np.corrcoef(linewidths,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,18].T)))[0][1])
    correlation_quiet_ori_res=np.corrcoef(np.ndarray.flatten(linew[0:200,300:500]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[0:200,300:500])))
    correlation_quiet_box_original_res.append(correlation_quiet_ori_res[0][1])
    correlation_network_ori_res=np.corrcoef(np.ndarray.flatten(linew[140:340,170:370]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[140:340,170:370])))
    correlation_network_box_original_res.append(correlation_network_ori_res[0][1])

    '''
    For data convolved with gaussian kernel
    '''
    correlation_calc=np.corrcoef(np.ndarray.flatten(linew_Lowres),np.ndarray.flatten(ART_conv))
    correlation.append(correlation_calc[0][1])
    correlation_quiet=np.corrcoef(np.ndarray.flatten(linew_Lowres[0:200,300:500]),np.ndarray.flatten(ART_conv[0:200,300:500]))
    correlation_quiet_box.append(correlation_quiet[0][1])
    correlation_network_b=np.corrcoef(np.ndarray.flatten(linew_Lowres[140:340,170:370]),np.ndarray.flatten(ART_conv[140:340,170:370]))
    correlation_network_box.append(correlation_network_b[0][1])
    lambd.append(waves[i])
    
    m, b = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T/1000)),linewidths , 1)
    slope_original_res.append(m)
    m_QS, b_QS = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[0:200,300:500]/1000)),np.ndarray.flatten(linew[0:200,300:500]) , 1)
    slope_quiet_box_original_res.append(m_QS)
    m_EN, b_EN = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[160:360,140:340]/1000)),np.ndarray.flatten(linew[160:360,140:340]) , 1)
    slope_network_box_original_res.append(m_EN)
    m_LR, b_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv)),np.ndarray.flatten(linew_Lowres) , 1)
    slope.append(m_LR)
    m_QS_LR, b_QS_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[0:200,300:500])),np.ndarray.flatten(linew_Lowres[0:200,300:500]) , 1)
    slope_quiet_box.append(m_QS_LR)
    m_EN_LR, b_EN_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]) , 1)
    slope_network_box.append(m_EN_LR)
    
    #writer_corr.writerow([waves[i],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])
    
    #writer_mb.writerow([waves[i],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    print('slopes intercepts lihile')
    
    print(i)


correlation=np.asarray(correlation).astype(np.float)
correlation_quiet_box=np.asarray(correlation_quiet_box).astype(np.float)
correlation_network_box=np.asarray(correlation_network_box).astype(np.float)
correlation_network_box_original_res=np.asarray(correlation_network_box_original_res).astype(np.float)
correlation_original_res=np.asarray(correlation_original_res).astype(np.float)
correlation_quiet_box_original_res=np.asarray(correlation_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (12,10)

plt.plot(lambd[1:],correlation_original_res[1:],marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd[1:],correlation_quiet_box_original_res[1:],marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd[1:],correlation_network_box_original_res[1:],marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd[1:],correlation[1:],marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd[1:],correlation_quiet_box[1:],marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd[1:],correlation_network_box[1:],marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.84,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=2.2,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('H alpha mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_correlations_Q_EN_22mm_all_res.png')
plt.close()
print('correlations cha plot kela')

slope=np.asarray(slope).astype(np.float)
slope_quiet_box=np.asarray(slope_quiet_box).astype(np.float)
slope_network_box=np.asarray(slope_network_box).astype(np.float)
slope_network_box_original_res=np.asarray(slope_network_box_original_res).astype(np.float)
slope_original_res=np.asarray(slope_original_res).astype(np.float)
slope_quiet_box_original_res=np.asarray(slope_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (13,10)



plt.plot(lambd[1:],slope_original_res[1:],marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd[1:],slope_quiet_box_original_res[1:],marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd[1:],slope_network_box_original_res[1:],marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd[1:],slope[1:],marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd[1:],slope_quiet_box[1:],marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd[1:],slope_network_box[1:],marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')
plt.axhline(y=0.0612,linestyle='--',color='k')
plt.axvline(x=3.0,linestyle='--',color='k')
#plt.axvline(x=3.6,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
#seaborn.regplot(waves, correl, color='k', marker='+')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.title('H alpha mm slopes',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_slopes_Q_EN_22mm_all_res.png')
plt.close()
#print('correlations cha plot kela')
print('slopes cha plot kela')


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
for i in range(1,35):
    
    bmaj_obs,bmin_obs,bpan_obs = Cal_FWHM(waves[i]),Cal_FWHM(waves[i]),0.00
    beam_kernel = beam_kernel_calulator(bmaj_obs,bmin_obs,bpan_obs,ART_pxsz)

    print("Kernel banavla.")    
    ART_conv=convolve((data_in_K[0,:,:,22].T)/1000,beam_kernel)
    print('mm data convolve kela')
    
    linew_Lowres=np.load(flist_Ha[i-1])
    
    '''
    for original resolution data
    '''
    correlation_original_resolution=np.corrcoef(linewidths,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,22].T)))
    correlation_original_res.append(np.corrcoef(linewidths,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,22].T)))[0][1])
    correlation_quiet_ori_res=np.corrcoef(np.ndarray.flatten(linew[0:200,300:500]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,22].T[0:200,300:500])))
    correlation_quiet_box_original_res.append(correlation_quiet_ori_res[0][1])
    correlation_network_ori_res=np.corrcoef(np.ndarray.flatten(linew[140:340,170:370]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,22].T[140:340,170:370])))
    correlation_network_box_original_res.append(correlation_network_ori_res[0][1])

    '''
    For data convolved with gaussian kernel
    '''
    correlation_calc=np.corrcoef(np.ndarray.flatten(linew_Lowres),np.ndarray.flatten(ART_conv))
    correlation.append(correlation_calc[0][1])
    correlation_quiet=np.corrcoef(np.ndarray.flatten(linew_Lowres[0:200,300:500]),np.ndarray.flatten(ART_conv[0:200,300:500]))
    correlation_quiet_box.append(correlation_quiet[0][1])
    correlation_network_b=np.corrcoef(np.ndarray.flatten(linew_Lowres[140:340,170:370]),np.ndarray.flatten(ART_conv[140:340,170:370]))
    correlation_network_box.append(correlation_network_b[0][1])
    lambd.append(waves[i])
    
    m, b = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T/1000)),linewidths , 1)
    slope_original_res.append(m)
    m_QS, b_QS = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[0:200,300:500]/1000)),np.ndarray.flatten(linew[0:200,300:500]) , 1)
    slope_quiet_box_original_res.append(m_QS)
    m_EN, b_EN = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,18].T[160:360,140:340]/1000)),np.ndarray.flatten(linew[160:360,140:340]) , 1)
    slope_network_box_original_res.append(m_EN)
    m_LR, b_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv)),np.ndarray.flatten(linew_Lowres) , 1)
    slope.append(m_LR)
    m_QS_LR, b_QS_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[0:200,300:500])),np.ndarray.flatten(linew_Lowres[0:200,300:500]) , 1)
    slope_quiet_box.append(m_QS_LR)
    m_EN_LR, b_EN_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]) , 1)
    slope_network_box.append(m_EN_LR)
    
    #writer_corr.writerow([waves[i],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])
    
    #writer_mb.writerow([waves[i],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    print('slopes intercepts lihile')
    
    print(i)


correlation=np.asarray(correlation).astype(np.float)
correlation_quiet_box=np.asarray(correlation_quiet_box).astype(np.float)
correlation_network_box=np.asarray(correlation_network_box).astype(np.float)
correlation_network_box_original_res=np.asarray(correlation_network_box_original_res).astype(np.float)
correlation_original_res=np.asarray(correlation_original_res).astype(np.float)
correlation_quiet_box_original_res=np.asarray(correlation_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (12,10)

plt.plot(lambd[1:],correlation_original_res[1:],marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd[1:],correlation_quiet_box_original_res[1:],marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd[1:],correlation_network_box_original_res[1:],marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd[1:],correlation[1:],marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd[1:],correlation_quiet_box[1:],marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd[1:],correlation_network_box[1:],marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

plt.axhline(y=0.84,linestyle='--',color='k')
plt.axvline(x=3.3,linestyle='--',color='k')
plt.axvline(x=2.8,linestyle='--',color='k')
plt.axvline(x=2.2,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('H alpha mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_correlations_Q_EN_3mm_all_res.png')
plt.close()
print('correlations cha plot kela')

slope=np.asarray(slope).astype(np.float)
slope_quiet_box=np.asarray(slope_quiet_box).astype(np.float)
slope_network_box=np.asarray(slope_network_box).astype(np.float)
slope_network_box_original_res=np.asarray(slope_network_box_original_res).astype(np.float)
slope_original_res=np.asarray(slope_original_res).astype(np.float)
slope_quiet_box_original_res=np.asarray(slope_quiet_box_original_res).astype(np.float)

plt.rcParams["figure.figsize"] = (13,10)



plt.plot(lambd[1:],slope_original_res[1:],marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
plt.plot(lambd[1:],slope_quiet_box_original_res[1:],marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
plt.plot(lambd[1:],slope_network_box_original_res[1:],marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd[1:],slope[1:],marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd[1:],slope_quiet_box[1:],marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd[1:],slope_network_box[1:],marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')
plt.axhline(y=0.0612,linestyle='--',color='k')
plt.axvline(x=3.0,linestyle='--',color='k')
#plt.axvline(x=3.6,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
#seaborn.regplot(waves, correl, color='k', marker='+')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.title('H alpha mm slopes',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_slopes_Q_EN_3mm_all_res.png')
plt.close()
#print('correlations cha plot kela')
print('slopes cha plot kela')




