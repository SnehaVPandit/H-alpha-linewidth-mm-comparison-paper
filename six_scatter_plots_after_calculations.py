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


FALC_temp_Kk=[4.7960528839455545, 4.904565764757657, 5.0189802544339726, 5.134869498466686, 5.249292223813534, 5.4668215687149955, 5.664036096037301, 5.839760804309665, 5.912732768602332, 5.9201922132928395, 5.927690234660785, 5.958077972899645, 5.965774232858892, 5.973511078558606, 5.994985358059721, 6.131961256112479, 6.2536928249424495, 6.363122805092185, 6.367267224968468, 6.377340716077308, 6.387487487267201, 6.4500059692136755, 6.460718965486915, 6.462820431285909, 6.471521605452283, 6.554868709756656, 6.72204671596397, 6.873316174831656, 7.01365015527365, 7.146061145814582, 7.272370751130562, 7.393708058092946, 7.496900195092534, 7.510836063784689, 7.527063829930758, 7.624308621957074, 7.692382669816156, 7.728730019227435, 7.734562942174342, 7.7663196569544155, 7.78859892256379, 7.841960576753278, 7.946803612346678, 8.049335060131554, 8.248256400723394, 8.39264411298781, 9.061008830744036, 9.26487955257423, 9.459282443510983, 9.644350698991214, 9.987746905996516]
FALC_temp_Kk=np.asarray(FALC_temp_Kk)

VALC_temp_Kk=[4.619188256784621, 4.697378812210936, 4.790599751023235, 4.8946207410520595, 5.0023575694274465, 5.211996808213638, 5.404183021861865, 5.576497652099214, 5.647973900758123, 5.655261469819057, 5.662582705500971, 5.692201113904036, 5.699687589306678, 5.707205847028972, 5.728033548160276, 5.858933563100163, 5.9711015159675, 6.0673977218871284, 6.070951013293328, 6.079558447848521, 6.088186390493476, 6.140418637584522, 6.149209196513559, 6.1509285427760565, 6.158027158326286, 6.224549157508108, 6.351015525229792, 6.459884988980598, 6.558651935323959, 6.6517998328675745, 6.7421002105146455, 6.83116057986364, 6.909182649810732, 6.919897225519591, 6.932429399663793, 7.008809183472059, 7.063616793195352, 7.09333981841728, 7.098139536064461, 7.124416092673954, 7.142996769620793, 7.18798668189076, 7.278357857837843, 7.369217548190039, 7.552198396316754, 7.690461673110729, 8.404214964744622, 8.655285913182107, 8.91463187306109, 9.182625593750103, 9.744846911143842]
VALC_temp_Kk=np.asarray(VALC_temp_Kk)

FALC_wavelengths=[ 299917.966973,  349904.294803,  399890.622633,  449876.950463,
        499863.278293,  599835.933953,  699808.589612,  799781.245272,
        845207.25843 ,  850001.447133,  854850.182915,  874811.723071,
        879948.506067,  885146.084435,  899753.900931,  999726.55659 ,
       1099699.21225 , 1199671.867909, 1203656.652998, 1213402.737266,
       1223308.152955, 1286311.045517, 1297447.999358, 1299644.523568,
       1308779.400013, 1399617.179228, 1599562.490546, 1799507.801864,
       1999453.113183, 2199398.424501, 2399343.73582 , 2599289.047138,
       2775096.960474, 2799234.358456, 2827457.388944, 2999179.669775,
       3121984.330465, 3188409.411996, 3199124.981093, 3257722.70355 ,
       3299097.636752, 3399070.292412, 3599015.60373 , 3798960.915048,
       4198851.537685, 4498769.504663, 5998359.33955 , 6498222.617846,
       6998085.896142, 7497949.174438, 8497675.73103 ]
FALC_wavelengths=np.asarray(FALC_wavelengths)

FALC_Ha_linewidth=0.9801971371996387#1.0207134534130091 #0.99#

VALC_Ha_linewidth=0.991659091080237

alf_cen_A_T=6.1161 #7.278543530676478 #6.3108 #
alf_cen_A_linew=0.9491098774878992
alfcen_A_T_err=0.545
alfcen_A_linew_err=0.0005317792465575621

for i in range(15,16):
    
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
    plt.plot(FALC_temp_Kk[26],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='FAL_C')
    plt.plot(VALC_temp_Kk[26],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='g',label='VAL_C')
    plt.plot(alf_cen_A_T, alf_cen_A_linew,linestyle='None', marker='o',markersize=15,color='k',label=r'$\alpha$ cen A')
    plt.errorbar(alf_cen_A_T, alf_cen_A_linew, yerr=alfcen_A_linew_err, xerr=alfcen_A_T_err, ecolor='k', elinewidth=1.0)
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
    plt.plot(FALC_temp_Kk[26],FALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='r',label='FAL_C')
    plt.plot(VALC_temp_Kk[26],VALC_Ha_linewidth,linestyle='None', marker='*',markersize=20,color='g',label='VAL_C')
    plt.plot(alf_cen_A_T, alf_cen_A_linew,linestyle='None', marker='o',markersize=15,color='k',label=r'$\alpha$ cen A')
    plt.errorbar(alf_cen_A_T, alf_cen_A_linew, yerr=alfcen_A_linew_err, xerr=alfcen_A_T_err, ecolor='k', elinewidth=1.0)
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
    plt.savefig('Ha_linewidth_mm_scatter_plot_Both_res_6_plot_FAL_VAL_alfcenA_%s.png' %np.around(waves[i], decimals=2))
    plt.close()
    
    print('6 scatter plot line saha plot kelay.')