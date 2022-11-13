
import warnings
warnings.filterwarnings('ignore')

###NUMPY
import numpy as np
import numpy.ma as ma
#import salat

###Dask 
import dask
from dask.distributed import Client, LocalCluster
from dask import delayed

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
import h5py
from os.path import splitext
import xarray as xr



@jit(nopython=True)
def fwhm_jit(cube, line, nx, ny, nwave, blue_w, red_w):
    '''
    See docstring in class Spectral_features.
    -------------------
    Args (new):
        nx (int)  :  number of pixels in x-direction
        ny (int)  :  number of pixels in y-direction
        blue_w (2D ndarray)  :  ndarray for calculating the wavelength
            in the blue wing of the line corresponding to the width
            calculated for the respective pixels.
        red_w (2D ndarray)  :  ndarray for calculating the wavelength in
            the red wing of the line corresponding to the width
            calculated for the respective pixels.
            
    cube: input with spectral lines
    line: wavelength grid
    nx,ny : no of pixels in x and y directions
    nwave: no of wavelength points. length of (line)
    blue_w,red_w: interpolated wavelength values for calculation of linewidth.
    '''
    #nwave_half = 60 # =np.argmin(emergent_intensity[i,j,:])
    for i in range(nx):
        for j in range(ny):
            nwave_half =np.argmin(cube[i,j])
            # defining max and min value for line in pixel (i, j) and
            # from that defining fwcm
            local_max = np.max(cube[i, j])
            local_min = np.min(cube[i, j])
            hm = (local_max + local_min) / 2.0 #half maxima
            #hm = (hm + local_min) / 2.0 #quarter maxima
            # hm2 = (hm + local_min) / 2.0
            # hm = (hm + hm2) / 2.0
            for k in range(1, nwave_half):
                # finding wavelength at fwcm in blue wing
                if ((cube[i, j, k] < hm) and (cube[i, j, k - 1] >= hm))\
                    or ((cube[i, j, k] > hm) and (cube[i, j, k - 1] <= hm)):
                    # interpolation: lam_hm = lam_prev +
                    # dlam * (fwhm - I_prev) / (I_current - I_prev)
                    blue_w[i, j] = line[k - 1]\
                        + (
                          (line[k] - line[k - 1]) *
                          ((hm - cube[i, j, k - 1]) /
                          (cube[i, j, k] - cube[i, j, k - 1]))
                          )
                    break
            for k in range(nwave_half, nwave - 1):
                # finding wavelength at fwcm in red wing
                if ((cube[i, j, k] > hm) and (cube[i, j, k - 1] <= hm))\
                    or ((cube[i, j, k] < hm) and (cube[i, j, k - 1] >= hm)):
                    # interpolation: lam_hm = lam_prev +
                    # dlam * (fwhm - I_prev) / (I_current - I_prev)
                    red_w[i, j] = line[k - 1]\
                        + (
                          (line[k] - line[k - 1]) *
                          ((hm - cube[i, j, k - 1]) /
                          (cube[i, j, k] - cube[i, j, k - 1]))
                          )
                    break
    return blue_w, red_w

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
    
def a1line(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--',color='k',linewidth=2.0, label="Molnar et.al.")

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

ART_pxsz = 0.066


with open(r"/mn/stornext/u3/snehap/rh/rh15d/run/output/H_alpha_linewidth_data.txt") as datFile:
    Linewidth=np.asarray([data.split(',')[4] for data in datFile])

linewidths_ori=Linewidth[1:].astype(np.float)

linew=linewidths_ori.reshape((504,504))
max_values=np.where(linew>200)
for i in range(len(max_values[0])):
    linew[max_values[0][i],max_values[1][i]]=0.0
linewidths_ori=np.ndarray.flatten(linew)

list_2 = ["wavelength",'original resolution slope','Original Quiet slope','Original Network slope','ALMA res full box slope','ALMA res Quiet sun box slope','ALMA res Network region box slope','original resolution intercept','Original Quiet intercept','Original Network intercept','ALMA res full box intercept','ALMA res Quiet sun box intercept','ALMA res Network region box intercept']
file_slope = open("slopes_intercepts_Ha_mm_Q_EN.txt", "w")
writer_mb = csv.writer(file_slope)
writer_mb.writerow(list_2)

list_1=["wavelength",'original resolution','Original Quiet','Original Network','ALMA res full box','ALMA res Quiet sun box','ALMA res Network region box']
file = open("correlations_Ha_mm_Q_EN.txt", "w")
writer_corr = csv.writer(file)
writer_corr.writerow(list_1)

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
    print(i)
    
    #file = open("H_alpha_convolved_Molnar_linewing_data_%s.txt" %waves[i], "w")
    #writer2 = csv.writer(file)
    #writer2.writerow(list_2)

    file = open("H_alpha_convolved_Molnar_linewidth_data_%s.txt" %waves[i], "w")
    writer = csv.writer(file)
    writer.writerow(list_1)
    
    bmaj_obs,bmin_obs,bpan_obs = Cal_FWHM(waves[i]),Cal_FWHM(waves[i]),0.00

    beam_kernel = beam_kernel_calulator(bmaj_obs,bmin_obs,bpan_obs,ART_pxsz)

    print("Kernel banavla.")

    delayed_results = np.empty([504,504,101],dtype=float)

    for j in range(101):
        conv_results = convolve(emergent_intensity[:,:,j],beam_kernel)
        delayed_results[:,:,j]=conv_results[:,:]
        #print(j)

    print("Convolution karunn zala")
    
    np.save("./BIFROST_H_alpha_Molnar_LOWRES_%s.npy" %waves[i],np.array(delayed_results),allow_pickle=True)
    print("poorna h alpha chi npy file lihili")

    blue_wing=np.empty([s,s],dtype=float)
    red_wing=np.empty([s,s],dtype=float)
    num_indices=np.shape(data.d.l)[0]
    blue_wing, red_wing=fwhm_jit(delayed_results, data.d.l, s, s, num_indices, blue_wing, red_wing)

    print("blue red wings zale calculate karun")

    print(np.max(blue_wing),np.min(blue_wing),np.max(red_wing),np.min(red_wing))
    
    plt.rcParams["figure.figsize"] = (13,10)
    plt.imshow(red_wing, origin='lower',cmap='gist_yarg',vmin=np.nanpercentile((red_wing).flatten(),1),vmax=np.nanpercentile((red_wing).flatten(),99))
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('H alpha red wing _%s' %waves[i],fontsize=25)
    cbar.set_label('Calculated H alpha red wing (Angstrom)', rotation=90, fontsize=20)
    cbar.ax.tick_params(labelsize=25)
    plt.savefig('Ha_red_wing_map_%s.png' %waves[i])
    plt.close()

    plt.rcParams["figure.figsize"] = (13,10)
    plt.imshow(blue_wing, origin='lower',cmap='gist_yarg',vmin=np.nanpercentile((blue_wing).flatten(),1),vmax=np.nanpercentile((blue_wing).flatten(),99))
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('H alpha blue wing %s' %waves[i],fontsize=25)
    cbar.set_label('Calculated H alpha blue wing (Angstrom)', rotation=90, fontsize=20)
    cbar.ax.tick_params(labelsize=25)
    plt.savefig('Ha_blue_wing_map_%s.png' %waves[i])
    plt.close()
    
    print('ani ata plot karun zale')
    
    linewidths=np.empty([s,s],dtype=float)
    for k in range(s):
        for j in range(s):
            linewidths[k,j]=np.abs(red_wing[k,j]-blue_wing[k,j])
            writer.writerow([x[k], y[j], blue_wing[k,j], red_wing[k,j], linewidths[k,j]])

    print('linewidths calculate kelya ani lihilya')
    
    max_values=np.where(linewidths>200)
    for k in range(len(max_values[0])):
        linewidths[max_values[0][k],max_values[1][k]]=np.nanmedian(linewidths)
    min_values=np.where(linewidths<200)
    for k in range(len(max_values[0])):
        linewidths[min_values[0][k],min_values[1][k]]=np.nanmedian(linewidths)
        
    linewidths[np.isnan(linewidths)] = np.nanmedian(linewidths)
    linew_Lowres=linewidths
    
    np.save("./BIFROST_H_alpha_linewidths_Molnar_LOWRES_%s.npy" %waves[i],np.array(linewidths),allow_pickle=True)
    print("linewidths chi npy file lihili")

    plt.rcParams["figure.figsize"] = (13,10)
    plt.imshow(linewidths.T, origin='lower',cmap='gray',vmin=np.nanpercentile((linewidths).flatten(),1),vmax=np.nanpercentile((linewidths).flatten(),99))
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('H alpha line width %s' %waves[i],fontsize=25)
    cbar.set_label('Calculated H alpha line width (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.savefig('Ha_linewidth_map_convolved_Molnar_%s.png' %waves[i])
    plt.close()

    print('ani plot kelya')
    
    ART_conv=convolve((data_in_K[0,:,:,i].T)/1000,beam_kernel)
    print('mm data convolve kela')
    
    np.save("./BIFROST_mm_Molnar_LOWRES_%s.npy" %waves[i],np.array(ART_conv),allow_pickle=True)
    print("mm chi npy file lihili")
    
    '''
    #for original resolution data
    '''
    correlation_original_resolution=np.corrcoef(linewidths_ori,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,i].T)))
    correlation_original_res.append(np.corrcoef(linewidths_ori,np.ndarray.flatten(np.asarray(data_in_K[:,:,:,i].T)))[0][1])
    correlation_quiet_ori_res=np.corrcoef(np.ndarray.flatten(linew[0:200,300:500]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,i].T[0:200,300:500])))
    correlation_quiet_box_original_res.append(correlation_quiet_ori_res[0][1])
    correlation_network_ori_res=np.corrcoef(np.ndarray.flatten(linew[140:340,170:370]),np.ndarray.flatten(np.asarray(data_in_K[0,:,:,i].T[140:340,170:370])))
    correlation_network_box_original_res.append(correlation_network_ori_res[0][1])

    '''
    #For data convolved with gaussian kernel
    '''
    linew_Lowres=linewidths
    
    correlation_calc=np.corrcoef(np.ndarray.flatten(linew_Lowres),np.ndarray.flatten(ART_conv))
    correlation.append(correlation_calc[0][1])
    correlation_quiet=np.corrcoef(np.ndarray.flatten(linew_Lowres[0:200,300:500]),np.ndarray.flatten(ART_conv[0:200,300:500]))
    correlation_quiet_box.append(correlation_quiet[0][1])
    correlation_network_b=np.corrcoef(np.ndarray.flatten(linew_Lowres[140:340,170:370]),np.ndarray.flatten(ART_conv[140:340,170:370]))
    correlation_network_box.append(correlation_network_b[0][1])
    lambd.append(waves[i])
    
    plt.rcParams['figure.figsize'] = [37,20]

    plt.subplot(2,3,1)
    plt.imshow(linew.T,origin='lower',vmin=np.nanpercentile((linewidths_ori),1),vmax=np.nanpercentile((linewidths_ori),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.title('H alpha linewidth (Angstrom)',fontsize=25)
    cbar.set_label('linewidth (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,2)
    plt.imshow(data_in_K[0,:,:,i]/1000,origin='lower',cmap='gray',vmin=np.nanpercentile((data_in_K[0,:,:,i].T/1000).flatten(),1),vmax=np.nanpercentile((data_in_K[0,:,:,i].T/1000).flatten(),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    plt.title('ART output %s mm' %np.around(waves[i], decimals=2),fontsize=25)
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
    #plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,3)
    counts,ybins,xbins,image = plt.hist2d(np.ndarray.flatten(data_in_K[0,:,:,i].T/1000),np.ndarray.flatten(np.asarray(linew)),bins=300,cmap='Reds')
    cbar=plt.colorbar()
    cbar.ax.tick_params(labelsize=25)
    m, b = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,i].T/1000)),np.ndarray.flatten(np.asarray(linew)) , 1)
    #slope_original_res.append(m*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,i].T[0:200,300:500]/1000),np.ndarray.flatten(linew[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_QS, b_QS = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,i].T[0:200,300:500]/1000)),np.ndarray.flatten(linew[0:200,300:500]) , 1)
    
    #slope_quiet_box_original_res.append(m_QS*1000)
    sns.kdeplot(np.ndarray.flatten(data_in_K[0,:,:,i].T[160:360,140:340]/1000),np.ndarray.flatten(linew[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_EN, b_EN = np.polyfit(np.ndarray.flatten(np.asarray(data_in_K[0,:,:,i].T[160:360,140:340]/1000)),np.ndarray.flatten(linew[160:360,140:340]) , 1)
    #slope_network_box_original_res.append(m_EN*1000)
    abline_EN(m_EN,b_EN)
    abline_QS(m_QS,b_QS)
    abline(m,b)
    a1line(6.12*10**-2,0.533)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    plt.title('Ha linewidth vs ALMA %s mm' %np.around(waves[i], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.xlim(4.7,11.7)
    plt.ylim(0.85,1.35)
    #plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('H alpha linewidth (Angstroms)',fontsize=25)
    
    plt.subplot(2,3,4)
    plt.imshow(linew_Lowres.T,origin='lower',vmin=np.nanpercentile((linew_Lowres),1),vmax=np.nanpercentile((linew_Lowres),99),cmap='gray')
    cbar=plt.colorbar()
    plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0,24, step=4),fontsize=25)
    plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
    plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
    #plt.title('H alpha linewidth (Angstrom)',fontsize=25)
    cbar.set_label('linewidth (Angstrom)', rotation=90, fontsize=30)
    cbar.ax.tick_params(labelsize=25)
    plt.gca().add_patch(Rectangle((304,4),200,200,linewidth=2,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((144,164),200,200,linewidth=2,edgecolor='k',facecolor='none'))

    plt.subplot(2,3,5)
    plt.imshow(ART_conv.T,origin='lower',cmap='gray',vmin=np.nanpercentile((ART_conv).flatten(),1),vmax=np.nanpercentile((ART_conv).flatten(),99))
    cbar=plt.colorbar()
    cbar.set_label('Brightness temperature (kK)',  fontsize=25)
    cbar.ax.tick_params(labelsize=25)
    #plt.title('ART output %s mm' %np.around(waves[i], decimals=2),fontsize=25)
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
    #slope.append(m_LR)
    sns.kdeplot(np.ndarray.flatten(ART_conv[0:200,300:500]),np.ndarray.flatten(linew_Lowres[0:200,300:500]),color='g',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_QS_LR, b_QS_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[0:200,300:500])),np.ndarray.flatten(linew_Lowres[0:200,300:500]) , 1)
    sns.kdeplot(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]),color='b',linestyles='--', linewidth=0.5,levels=5, thresh=.2,fill=True, alpha=0.4)
    m_EN_LR, b_EN_LR = np.polyfit(np.ndarray.flatten(np.asarray(ART_conv[160:360,140:340])),np.ndarray.flatten(linew_Lowres[160:360,140:340]) , 1)
    #slope_network_box.append(m_EN_LR)
    abline(m_LR,b_LR)
    abline_QS(m_QS_LR,b_QS_LR)
    abline_EN(m_EN_LR,b_EN_LR)
    a1line(6.12*10**-2,0.533)
    #plt.plot(FALC_temp_Kk[15],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='k',label='FAL_C')
    #plt.plot(VALC_temp_Kk[15],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='VAL_C')
    plt.xticks(fontsize=25)
    #plt.title('Network region: Ha linewidth vs ALMA %s' %np.around(waves[i], decimals=2),fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel('ALMA Brightness Temperature(kK)',fontsize=25)
    plt.ylabel('H alpha linewidth (Angstroms)',fontsize=25)
    plt.legend(fontsize=25)
    plt.xlim(4.7,11.7)
    plt.ylim(0.85,1.35)
    plt.tight_layout()
    #plt.show()
    plt.savefig('Ha_linewidth_mm_scatter_plot_Both_res_6_plot_%s.png' %np.around(waves[i], decimals=2))
    plt.close()
    
    print('6 scatter plot line saha plot kelay.')

    print([waves[i],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])
    writer_corr.writerow([waves[i],correlation_original_resolution[0][1],correlation_quiet_ori_res[0][1],correlation_network_ori_res[0][1],correlation_calc[0][1],correlation_quiet[0][1],correlation_network_b[0][1]])

    print([waves[i],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    writer_mb.writerow([waves[i],m,m_QS,m_EN,m_LR,m_QS_LR,m_EN_LR,b,b_QS,b_EN,b_LR,b_QS_LR,b_EN_LR])
    print('slopes ani intercepts ani correlations lihile')



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
plt.title('H alpha mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('correlation coefficient',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_correlations_Q_EN_plot_previous _EN_box.png')
plt.close()
print('correlations cha plot kela')

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
plt.title('H alpha mm correlations',fontsize=25)
plt.xlabel('ALMA wavelengths (mm)',fontsize=25)
plt.ylabel('slopes',fontsize=25)
plt.savefig('mm_bands_Halpha_flux_slopes_Q_EN_plot.png')
#plt.close()
#print('correlations cha plot kela')
print('slopes cha plot kela')


