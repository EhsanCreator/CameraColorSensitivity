import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
Data=np.array(pd.read_excel('.\Cameras 400-700.xlsx'))
OBS=np.array(pd.read_excel('.\OBS 400-700.xlsx'))

#----------------------------- CAMERA NAMES -------------------------

camera=['Canon 1DMarkIII','Canon 20D','Canon 300D',' Canon 40D',
        'Canon 500D','Canon 50D','Canon 5DMark II','Canon 600D',
        'Canon 60D','Hasselblad H2','Nikon D3X','Nikon D200',
        'Nikon D3','Nikon D300s','Nikon D40','Nikon D50',
        'Nikon D5100','Nikon D700','Nikon D80','Nikon D90',
        'Nokia N900','Olympus E-PL2','Pentax K-5','Pentax Q',
        'Point Grey Grasshopper 50S5C','Point Grey Grasshopper2 14S5C',
        'Phase One','SONY NEX-5N']
camera=np.array(camera)

#------------------------------ CALCULATING --------------------------

RGB_Bar=OBS[:,(6,7,8)]
R_Bar=OBS[:,6]
G_Bar=OBS[:,7]
B_Bar=OBS[:,8]


g = []
r = []
for i in range(1,84,3):
    temp1 = []
    temp2 = []
    for j in range(1,32):
        temp1.append(Data[j,i]/(Data[j,i]+Data[j,i+1]+Data[j,i+2]))
        temp2.append(Data[j,i+1]/(Data[j,i]+Data[j,i+1]+Data[j,i+2]))
    r.append(temp1)
    g.append(temp2)
r = np.array(r)
g = np.array(g)

rr = []
gg = []
for i in range(0,28):
    rr.append(np.hstack((r[i], r[i,0])))
    gg.append(np.hstack((g[i], g[i,0])))
rr = np.array(rr)
gg = np.array(gg)


RGB_rgb = []
for i in range (1,84,3):
    RGB_rgb.append(np.hstack((RGB_Bar,Data[1:,i:i+3])))

RGBs = np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar))

As=[]
for i in range (1,84,3):
    As.append(np.dot((Data[1:,i:i+3]).astype(float),np.linalg.pinv((Data[1:,i:i+3]).astype(float))))

cal = [np.dot(RGBs,As[i]) for i in range (0,28)]
Vora = [np.trace(cal[i]) for i in range (0,28)]
Vora.append(np.trace(np.dot(np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)),np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)))))

Vora = [Vora[i]/3 for i in range (0,29)]


RGB_rgb.append(np.hstack((RGB_Bar,RGB_Bar)))

uvw = []; u = []; v = []; w = []
for i in range (0,29):
    uvw.append(np.linalg.svd(RGB_rgb[i].astype(float)))
    u.append(uvw[i][0])
    v.append((uvw[i][1]).round(4))
    w.append(uvw[i][2])

sums = []
for i in range (0,29):
    sums.append((sum(v[i][0:3])/sum(v[i][0:6])).round(4))
    
sums2 = []
for i in range (0,29):
    sums2.append((sum(v[i][3:5])/sum(v[i][0:6])).round(4))


Residual = [(1 - sums[i]).round(4) for i in range(0,29)]
Residual_percent = [(Residual[i]*100).round(4) for i in range(0,29)]
Residual2 = [v[i][3]/sum(v[i][0:6]) for i in range(0,29)]
corr = np.corrcoef((Residual,Vora))*100
corr2 = np.corrcoef((sums2,Vora))*100
corr3 = np.corrcoef((Residual2,Vora))*100

# ========================== PLOTING DATA ===========================

# plt.figure(figsize = (25,6))
# plt.subplot(1,3,1)
# plt.scatter(Residual,Vora,label=('corr',(corr[0,1]).round(2)))
# plt.xlabel('Residual',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# plt.title('Residual VS Vora',fontsize=15,fontweight="bold")
# plt.legend()
# for i in range(0,28):
#     plt.annotate(camera[i], xy=(Residual[i],Vora[i]),fontsize=1.5)
# plt.subplot(1,3,2)
# plt.scatter(sums2,Vora,label=('corr',(corr2[0,1]).round(2)))
# plt.xlabel('sums2',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# plt.title('sums2 VS Vora',fontsize=15,fontweight="bold")
# plt.legend()
# for i in range(0,28):
#     plt.annotate(camera[i], xy=(sums2[i],Vora[i]),fontsize=1.5)
# plt.subplot(1,3,3)
# plt.scatter(Residual2,Vora,label=('corr',(corr3[0,1]).round(2)))
# plt.xlabel('Residual2',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# plt.title('Residual2 VS Vora',fontsize=15,fontweight="bold")
# plt.legend()
# for i in range(0,28):
#     plt.annotate(camera[i], xy=(Residual2[i],Vora[i]),fontsize=1.5)
# plt.savefig('Figures4',dpi=500)
# plt.close()

#=======================================================================

counter = [i for i in range (0,29)]
plt.figure()
plt.scatter(Vora,counter)




area_camera=[]
for i in range(0,28):
    area_camera.append(np.trapz(gg[i],rr[i]).round(4)) 



