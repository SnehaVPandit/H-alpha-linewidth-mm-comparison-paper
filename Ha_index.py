import numpy as np
import matplotlib.pyplot as plt
from helita.sim import rh15d
import astropy.units as u
from helita.vis import rh15d_vis
from helita.sim import multi3d
import xarray as xr
import csv
list_1 = ["x", "y", "R", "V", "Ha", "V+R", "Ha_index"]

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

dat789 = xr.open_dataset('/mn/stornext/d18/RoCS/snehap/rh_first/rh15d/run_parallel_H/H_alpha_V_R_bands/output/output_ray.hdf5')

wave_789 = dat789.wavelength
indices_789 = np.arange(len(wave_789))

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

s=504  #size of the box in pixels

file = open("H_alpha_index_data.txt", "w")
writer = csv.writer(file)
writer.writerow(list_1)
x=[]
y=[]

for i in range(s):
    x.append(i)
    y.append(i)

Ha_index_789_all=np.empty([s,s],dtype=float)
Ha_789_all=np.empty([s,s],dtype=float)
R_789_all=np.empty([s,s],dtype=float)
V_789_all=np.empty([s,s],dtype=float)
V_R_all=np.empty([s,s],dtype=float)
'''
inti_Ha=0.00

for i in range(13,88):
    inti_Ha=inti_Ha+emergent_intensity[0,0,i]
inti_Ha=inti_Ha/(88-13)

inti_V=0.00

for i in range(11,34):
    inti_V=inti_V+dat789.intensity[0,0,i]
inti_V=inti_V/(34-11)

inti_R=0.00

for i in range(36,59):
    inti_R=inti_R+dat789.intensity[0,0,i]
inti_R=inti_R/(59-36)

Ha_index=inti_Ha/(inti_V+inti_R)

Ha_index
'''
for i in range(s):
    for j in range(s):
        inti_Ha=0.00

        for l in range(13,88):
            inti_Ha=inti_Ha+emergent_intensity[i,j,l]
        inti_Ha=inti_Ha/(88-13)

        inti_V=0.00

        for l in range(11,34):
            inti_V=inti_V+dat789.intensity[i,j,l]
        inti_V=inti_V/(34-11)
        
        inti_R=0.00

        for l in range(36,59):
            inti_R=inti_R+dat789.intensity[i,j,l]
        inti_R=inti_R/(59-36)

        V_789_all[i][j]=(inti_V)
        Ha_789_all[i][j]=(inti_Ha)
        R_789_all[i][j]=(inti_R)
        V_R_all[i][j]=(inti_V+inti_R)
        Ha_index_789_all[i][j]=(inti_Ha)/(inti_V+inti_R)
        writer.writerow([x[i], y[j], R_789_all[i][j], V_789_all[i][j], Ha_789_all[i][j],  V_R_all[i][j], Ha_index_789_all[i][j]])

file.close()

plt.rcParams["figure.figsize"] = (13,10)
#fig = plt.figure()
#ax = fig.add_subplot(1, 1, 1)
plt.imshow(Ha_index_789_all, origin='lower',cmap='gist_yarg',vmax=200)
cbar=plt.colorbar()
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
plt.title('H alpha index',fontsize=25)
#cbar=plt.colorbar()
cbar.set_label('Calculated H alpha index', rotation=90, fontsize=30)
cbar.ax.tick_params(labelsize=25)
#plt.show()
plt.savefig('Ha_index_map_truncated.png')
plt.close()


plt.rcParams["figure.figsize"] = (26,21)
plt.subplot(2, 2, 1)
plt.imshow(V_789_all*10**8, origin='lower',cmap='gray')
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
plt.title('V band',fontsize=25)
cbar=plt.colorbar()
cbar.set_label('Integrated flux (10E-8  W/(Hz m2 sr))', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.subplot(2, 2, 2)
plt.imshow(Ha_789_all*10**5, origin='lower',cmap='gray')
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
plt.title('H alpha line',fontsize=25)
cbar=plt.colorbar()
cbar.set_label('Integrated flux (10E-5  W/(Hz m2 sr))', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.subplot(2, 2, 3)
plt.imshow(R_789_all*10**8, origin='lower',cmap='gray')
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
plt.title('R band',fontsize=25)
cbar=plt.colorbar()
cbar.set_label('Integrated flux (10E-8  W/(Hz m2 sr))', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
plt.subplot(2, 2, 4)
plt.imshow(Ha_index_789_all, origin='lower',cmap='gray')
plt.xticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.yticks(ticks=np.arange(0, 504, step=84),labels=np.arange(0, 24, step=4),fontsize=25)
plt.xlabel('Horizontal X direction (Mm)',fontsize=25)
plt.ylabel('Horizontal Y direction (Mm)',fontsize=25)
plt.title('H alpha index',fontsize=25)
cbar=plt.colorbar()
cbar.set_label('Calculated Ha index)', rotation=90, fontsize=25)
cbar.ax.tick_params(labelsize=25)
#plt.show()
plt.savefig('Ha_index_components_4plots.png')
plt.close()
