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
#import tqd
import glob
from numba import jit
import csv

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

data = h5py.File("/mn/stornext/d18/RoCS/snehap/mm_data_from_BIFROST/bifrost_en024048_1000_int_sneha.h5",'r')
wave = data['Wavelength'][:] * u.angstrom
freq = wave.to(u.GHz,equivalencies=u.spectral())
wmm = wave.to(u.mm)
waves=wmm.value

flist_Ha = sorted(glob.glob('/mn/stornext/d18/RoCS/snehap/H_alpha_index_plots/Ha_index_LOWRES_*.npy'))
flist_S = sorted(glob.glob('/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/s_index_LOWRES_*.npy'))
flist_IRT = sorted(glob.glob('/mn/stornext/d18/RoCS/snehap/CaII_mm_comparison/IRT_index_LOWRES_*.npy'))
flist_linew = sorted(glob.glob('/mn/stornext//u3/snehap/alma_plots/ALMA_Molnar/Ha_linewidth_correct_scatter_plots/BIFROST_H_alpha_linewidths_Molnar_LOWRES_*.npy'))


list_mb = ["wavelength",'original resolution slope','Original Quiet slope','Original Network slope','ALMA res full box slope','ALMA res Quiet sun box slope','ALMA res Network region box slope','original resolution intercept','Original Quiet intercept','Original Network intercept','ALMA res full box intercept','ALMA res Quiet sun box intercept','ALMA res Network region box intercept']
file_slope = open("slopes_intercepts_CaII_IRT_Ha_index_mm_Q_EN.txt", "w")
writer_mb_Ha_index = csv.writer(file_slope)
writer_mb_Ha_index.writerow(list_mb)

list_corr=["wavelength",'original resolution','Original Quiet','Original Network','ALMA res full box','ALMA res Quiet sun box','ALMA res Network region box']
file = open("correlations_CaII_IRT_Ha_index_mm_Q_EN.txt", "w")
writer_corr_Ha_index = csv.writer(file)
writer_corr_Ha_index.writerow(list_corr)

#correlation_original_res=[]
#correlation_quiet_box_original_res=[]
#correlation_network_box_original_res=[]
correlation=[]
correlation_quiet_box=[]
correlation_network_box=[]

#slope_original_res=[]
#slope_quiet_box_original_res=[]
#slope_network_box_original_res=[]
slope=[]
slope_quiet_box=[]
slope_network_box=[]
lambd=[]

for wave_index in range(34):
    print(wave_index)
    
    #S_index=np.asarray(np.load(flist_S[wave_index]))
    IRT_index=np.asarray(np.load(flist_IRT[wave_index-1]))
    Ha_index=np.asarray(np.load(flist_Ha[wave_index]))
    #Ha_linewidths=np.asarray(np.load(flist_linew[wave_index-1]))
    
    index_1=IRT_index
    index_2=Ha_index

    '''
    #For data convolved with gaussian kernel
    '''    
    correlation_calc=np.corrcoef(np.ndarray.flatten(index_1),np.ndarray.flatten(index_2))
    correlation.append(correlation_calc[0][1])
    correlation_quiet=np.corrcoef(np.ndarray.flatten(index_1[0:200,300:500]),np.ndarray.flatten(index_2[0:200,300:500]))
    correlation_quiet_box.append(correlation_quiet[0][1])
    correlation_network_b=np.corrcoef(np.ndarray.flatten(index_1[140:340,170:370]),np.ndarray.flatten(index_2[140:340,170:370]))
    correlation_network_box.append(correlation_network_b[0][1])
    lambd.append(waves[wave_index+1])
    
    plt.rcParams['figure.figsize'] = [37,10]

    plt.subplot(1,3,1)
    plt.imshow(index_1,origin='lower',vmin=np.nanpercentile((np.ndarray.flatten(index_1)),1),vmax=np.nanpercentile((np.ndarray.flatten(index_1)),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('IRT index',fontsize=25)
    #cbar.set_label('linewidth (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(1,3,2)
    plt.imshow(index_2,origin='lower',cmap='gray',vmin=np.nanpercentile(np.ndarray.flatten(index_2),1),vmax=np.nanpercentile((np.ndarray.flatten(index_2)),99))
    cbar=plt.colorbar()
    #cbar.set_label('Ha_index',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    plt.title('Ha index' %np.around(waves[wave_index], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(1,3,3)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(index_2),np.ndarray.flatten(np.asarray(index_1)),bins=300,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m, b = np.polyfit(np.ndarray.flatten(np.asarray(index_2)),np.ndarray.flatten(np.asarray(index_1)) , 1)
    slope.append(m)
    sns.kdeplot(np.ndarray.flatten(index_2[0:200,300:500]),np.ndarray.flatten(index_1[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_QS, b_QS = np.polyfit(np.ndarray.flatten(np.asarray(index_2[0:200,300:500])),np.ndarray.flatten(index_1[0:200,300:500]) , 1)
    
    slope_quiet_box.append(m_QS)
    sns.kdeplot(np.ndarray.flatten(index_2[160:360,140:340]),np.ndarray.flatten(index_1[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_EN, b_EN = np.polyfit(np.ndarray.flatten(np.asarray(index_2[160:360,140:340])),np.ndarray.flatten(index_1[160:360,140:340]) , 1)
    slope_network_box.append(m_EN)
    abline_EN(m_EN,b_EN)
    abline_QS(m_QS,b_QS)
    abline(m,b)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    plt.title('IRT index vs Ha index at res: %s mm' %np.around(waves[wave_index+1], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    #plt.xlim(4.7,11.7)
    #plt.ylim(0.85,1.35)
    plt.xlabel('Ha index',fontsize=25)
    plt.ylabel('IRT index',fontsize=25)
    
    plt.savefig('Ha_index_IRT_index_scatter_plot_Both_res_6_plot_%s.png' %np.around(waves[wave_index+1], decimals=2))
    plt.close()
    
    print('IRT index vs Ha index cha 3 scatter plot line saha plot kelay.')

    print([waves[wave_index+1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])
    writer_corr_Ha_index.writerow([waves[wave_index+1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])

    print([waves[wave_index+1],m,m_QS,m_EN])
    writer_mb_Ha_index.writerow([waves[wave_index+1],m,m_QS,m_EN])
    print('IRT index vs Ha index che slopes ani intercepts ani correlations lihile')


# Ha index che slopes ani correlations

correlation=np.asarray(correlation).astype(np.float)
correlation_quiet_box=np.asarray(correlation_quiet_box).astype(np.float)
correlation_network_box=np.asarray(correlation_network_box).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

#plt.plot(lambd,correlation_original_res,marker='.',markersize=10,color='k',linestyle='-',linewidth=2,label='Original resolution')
#plt.plot(lambd,correlation_quiet_box_original_res,marker='.',markersize=10,color='k',linewidth=2,linestyle=':',label='QS Original')
#plt.plot(lambd,correlation_network_box_original_res,marker='.',markersize=10,color='k',linestyle='--',linewidth=2,label='EN original')
plt.plot(lambd,correlation,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
plt.plot(lambd,correlation_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
plt.plot(lambd,correlation_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')

#plt.axhline(y=0.84,linestyle='--',color='k')
#plt.axvline(x=3.3,linestyle='--',color='k')
#plt.axvline(x=2.8,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=25)
plt.title('IRT index vs Ha index correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('IRT_index_Ha_index_correlations_Q_EN_plot_.png')
plt.close()
print('Ha index correlations cha plot kela')

slope=np.asarray(slope).astype(np.float)
slope_quiet_box=np.asarray(slope_quiet_box).astype(np.float)
slope_network_box=np.asarray(slope_network_box).astype(np.float)

plt.rcParams["figure.figsize"] = (10,10)

#plt.plot(lambd,slope_original_res,marker='.',markersize=10,linewidth=2,label='Original resolution')
plt.plot(lambd,slope,marker='.',markersize=10,color='r',linestyle='-',linewidth=2,label='ALMA resolution')
#plt.plot(lambd,slope_quiet_box_original_res,marker='.',markersize=10,linewidth=2,label='Quiet sun Original')
plt.plot(lambd,slope_quiet_box,marker='.',markersize=10,color='r',linewidth=2,linestyle=':',label='QS ALMA res')
#plt.plot(lambd,slope_network_box_original_res,marker='.',markersize=10,linewidth=2,label='Network region original')
plt.plot(lambd,slope_network_box,marker='.',markersize=10,color='r',linestyle='--',linewidth=2,label='EN ALMA res')
#plt.axhline(y=0.0612,linestyle='--',color='k')
#plt.axvline(x=3.0,linestyle='--',color='k')
#plt.axvline(x=3.6,linestyle='--',color='k')
#plt.axvline(x=2.2,linestyle='--',color='r')
#seaborn.regplot(waves, correl, color='k', marker='+')
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize=20)
plt.title('IRT index vs Ha index mm slopes',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('IRT_index_Ha_index_slopes_Q_EN_plot.png')
#plt.close()
#print('correlations cha plot kela')
print('IRT index vs Ha index slopes cha plot kela')

print('zala sagla.')
