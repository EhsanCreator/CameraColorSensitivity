import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Data=np.array(pd.read_excel('.\Cameras.xlsx'))
OBS=np.array(pd.read_excel('.\OBS.xlsx'))

#--------------------------- CAMERA NAMES -------------------------

camera=['Canon 1DMarkIII','Canon 20D','Canon 300D',' Canon 40D',
        'Canon 500D','Canon 50D','Canon 5DMark II','Canon 600D',
        'Canon 60D','Hasselblad H2','Nikon D3X','Nikon D200',
        'Nikon D3','Nikon D300s','Nikon D40','Nikon D50',
        'Nikon D5100','Nikon D700','Nikon D80','Nikon D90',
        'Nokia N900','Olympus E-PL2','Pentax K-5','Pentax Q',
        'Point Grey Grasshopper 50S5C','Point Grey Grasshopper2 14S5C',
        'Phase One','SONY NEX-5N']
camera=np.array(camera)

#-------------------- CALCULATING X AND Y AXIS ---------------------

k_D65=100/(sum(OBS[:,2]*OBS[:,4]))
X_D65=k_D65*OBS[:,1]*OBS[:,4]
Y_D65=k_D65*OBS[:,2]*OBS[:,4]
Z_D65=k_D65*OBS[:,3]*OBS[:,4]

x_D65=(X_D65/(X_D65+Y_D65+Z_D65))
y_D65=(Y_D65/(X_D65+Y_D65+Z_D65))
x_D65 = np.hstack((x_D65, x_D65[0]))
y_D65 = np.hstack((y_D65, y_D65[0]))

R_Bar=OBS[:,6]
G_Bar=OBS[:,7]
B_Bar=OBS[:,8]
r_Bar=(R_Bar/(R_Bar+G_Bar+B_Bar))
g_Bar=(G_Bar/(R_Bar+G_Bar+B_Bar))
r_Bar=np.hstack((r_Bar,r_Bar[0]))
g_Bar=np.hstack((g_Bar,g_Bar[32]))
area_rgb=np.trapz(g_Bar,r_Bar).round(4)

R = []
G = []
B = []
for i in range (1,84,3):
    R.append(Data[1:, i])
    G.append(Data[1:, i+1])
    B.append(Data[1:, i+2])

Lambd= Data[1:,0]

legend_properties = {'weight':'bold', 'size': 15}

plt.figure(figsize=(18,25))
for i in range (28):
    plt.subplot(7, 4, i+1)
    plt.title(camera[i],fontsize=15, fontweight = 'bold')
    plt.plot(Lambd, R[i], color='red', label='R')
    plt.plot(Lambd, G[i], color='green', label='G')
    plt.plot(Lambd, B[i], color='blue', label='B')
    plt.xlabel('Wavelength (nm)', fontsize=15, fontweight = 'bold', loc='right')
    plt.ylabel('Relative value', fontsize=15, fontweight = 'bold', loc='top')
    plt.xlim(400,720)
    plt.legend()
    plt.xticks(size = 15)
    plt.yticks(size = 15)
    plt.tight_layout()
plt.savefig('all_of_rgb',dpi=300)
plt.close()
 




   
    























