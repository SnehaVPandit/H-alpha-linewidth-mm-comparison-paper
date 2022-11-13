import h5py
import numpy as np
import matplotlib as mpl
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

linewidths=Linewidth[1:].astype(np.float)

slopes=[]
lambd=[]

linew=linewidths.reshape((504,504))
max_values=np.where(linew>200)
for i in range(len(max_values[0])):
    linew[max_values[0][i],max_values[1][i]]=0.0
linewidths=np.ndarray.flatten(linew)

linewidths_ori=linew

waves=wmm.value
print(waves)

flist_Ha = sorted(glob.glob('/mn/stornext/u3/snehap/alma_plots/ALMA_Molnar/Ha_linewidth_correct_scatter_plots/BIFROST_H_alpha_linewidths_Molnar_LOWRES_*.npy'))

flist_mm = sorted(glob.glob('/mn/stornext/u3/snehap/alma_plots/ALMA_Molnar/Ha_linewidth_correct_scatter_plots/BIFROST_mm_Molnar_LOWRES_*.npy'))

FALC_temp_Kk=[4.796052884,4.904565765,5.018980254,5.134869498,5.249292224,5.466821569,5.664036096,5.839760804,5.994985358,6.131961256,6.253692825,6.363122805,6.462820431,6.55486871,6.722046716,6.873316175,7.013650155,7.146061146,7.272370751,7.393708058,7.510836064,7.624308622,7.734562942,7.788598923,7.841960577,7.946803612,8.04933506,8.248256401,8.392644113,9.061008831,9.264879553,9.459282444,9.644350699,9.987746905996516]
FALC_temp_Kk=np.asarray(FALC_temp_Kk)
VALC_temp_Kk=[4.619188257,4.697378812,4.790599751,4.894620741,5.002357569,5.211996808,5.404183022,5.576497652,5.728033548,5.858933563,5.971101516,6.067397722,6.150928543,6.224549158,6.351015525,6.459884989,6.558651935,6.651799833,6.742100211,6.83116058,6.919897226,7.008809183,7.098139536,7.14299677,7.187986682,7.278357858,7.369217548,7.552198396,7.690461673,8.404214965,8.655285913,8.914631873,9.182625594,9.744846911143842]
VALC_temp_Kk=np.asarray(VALC_temp_Kk)

FALC_wavelengths=[299917.967,349904.2948,399890.6226,449876.9505,499863.2783,599835.934,699808.5896,799781.2453,899753.9009,999726.5566,1099699.212,1199671.868,1299644.524,1399617.179,1599562.491,1799507.802,1999453.113,2199398.425,2399343.736,2599289.047,2799234.358,2999179.67,3199124.981,3299097.637,3399070.292,3599015.604,3798960.915,4198851.538,4498769.505,5998359.34,6498222.618,6998085.896,7497949.174,8497675.73103]
FALC_wavelengths=np.asarray(FALC_wavelengths)

FALC_Ha_linewidth=0.9801971371996387#1.0207134534130091 #0.99#

VALC_Ha_linewidth=0.991659091080237
'''
alf_cen_A_T=7.278543530676478 #6.3108 #6.1161 #
alf_cen_A_linew=0.9491098774878992
alfcen_A_T_err=0.545
alfcen_A_linew_err=0.0005317792465575621'''

for i in range(1,35):
    print(i)
    linew_Lowres=np.load(flist_Ha[i-1])
    ART_conv=np.load(flist_mm[i-1])
    
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
    plt.plot(np.nanmean(data_in_K[0,:,:,i].T/1000),np.nanmean(linew),linestyle='None', marker='o',markersize=20,color='r')
    plt.plot(np.nanmean(data_in_K[0,:,:,i].T[0:200,300:500]/1000),np.nanmean(linew[0:200,300:500]),linestyle='None', marker='o',markersize=20,color='g')
    plt.plot(np.nanmean(data_in_K[0,:,:,i].T[160:360,140:340]/1000),np.nanmean(linew[160:360,140:340]),linestyle='None', marker='o',markersize=20,color='b')
    plt.plot(FALC_temp_Kk[i-1],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='FAL_C')
    plt.plot(VALC_temp_Kk[i-1],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='g',label='VAL_C')
    #plt.plot(alf_cen_A_T, alf_cen_A_linew,linestyle='None', marker='o',markersize=15,color='k',label=r'$\alpha$ cen A')
    #plt.errorbar(alf_cen_A_T, alf_cen_A_linew, yerr=alfcen_A_linew_err, xerr=alfcen_A_T_err, ecolor='k', elinewidth=1.0)
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
    plt.plot(np.nanmean(ART_conv),np.nanmean(linew_Lowres),linestyle='None', marker='o',markersize=20,color='r')
    plt.plot(np.nanmean(ART_conv[0:200,300:500]),np.nanmean(linew_Lowres[0:200,300:500]),linestyle='None', marker='o',markersize=20,color='g')
    plt.plot(np.nanmean(ART_conv[160:360,140:340]),np.nanmean(linew_Lowres[160:360,140:340]),linestyle='None', marker='o',markersize=20,color='b')
    plt.plot(FALC_temp_Kk[i-1],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='FAL_C')
    plt.plot(VALC_temp_Kk[i-1],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='g',label='VAL_C')
    #plt.plot(alf_cen_A_T, alf_cen_A_linew,linestyle='None', marker='o',markersize=15,color='k',label=r'$\alpha$ cen A')
    #plt.errorbar(alf_cen_A_T, alf_cen_A_linew, yerr=alfcen_A_linew_err, xerr=alfcen_A_T_err, ecolor='k', elinewidth=1.0)
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
    plt.savefig('Ha_linewidth_mm_scatter_plot_Both_res_6_plot_FAL_VAL_same_axes_ranges_%s.png' %np.around(waves[i], decimals=2))
    plt.close()
    
    print('6 scatter plot line saha plot kelay.')
