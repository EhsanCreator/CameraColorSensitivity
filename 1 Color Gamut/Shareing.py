import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon


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
area_OBS=np.trapz(y_D65,x_D65).round(4)

g = []
r = []
for i in range(1,84,3):
    temp1 = []
    temp2 = []
    for j in range(1,34):
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

area_camera=[]
for i in range(0,28):
    area_camera.append(np.trapz(gg[i],rr[i]).round(4)) 

R_Bar=OBS[:,6]
G_Bar=OBS[:,7]
B_Bar=OBS[:,8]
r_Bar=(R_Bar/(R_Bar+G_Bar+B_Bar))
g_Bar=(G_Bar/(R_Bar+G_Bar+B_Bar))
r_Bar=np.hstack((r_Bar,r_Bar[0]))
g_Bar=np.hstack((g_Bar,g_Bar[32]))
area_rgb=np.trapz(g_Bar,r_Bar).round(4)


Lambd=np.hstack((Data[1:,0],400))

#_______________ CALCULATE CURVES  ________________

polys1=Polygon(list(zip(x_D65, y_D65)))
polys3=Polygon(list(zip(r_Bar, g_Bar)))

polys2=[]
for i in range(0,28):
    polys2.append(Polygon(list(zip(rr[i,:], gg[i,:]))))

df1 = gpd.GeoDataFrame({'geometry': polys1, 'df1':[1,2]})
df3 = gpd.GeoDataFrame({'geometry': polys3, 'df1':[1,2]})


res_bar=[]
res=[]
df2=[]
for i in range(0,28):
    df2.append(gpd.GeoDataFrame({'geometry': polys2[i], 'df2':[1,2]})) 
    res.append(gpd.overlay(df1,df2[i],how='intersection'))
    res_bar.append(gpd.overlay(df3,df2[i],how='intersection'))

ddff2=[df2[i].convex_hull for i in range(0,28)]


area_res=[]
for i in range(0,28):
    area_res.append((res[i].area).round(4))
area_res=np.delete(area_res,[0,1,2],1)
area_res=area_res.tolist()


area_res_bar=[]
for i in range(0,28):
    area_res_bar.append((res_bar[i].area).round(4))
area_res_bar=np.array(area_res_bar)  
area_res_bar=np.delete(area_res_bar,[0,1,2],1)
area_res_bar=area_res_bar.tolist()


def coord_lister(geom):
    coords = list(geom.exterior.coords)
    return (coords)

qaz=[]
for i in range(0,28):
    qaz.append(coord_lister(res[i].loc[0][2]))
    
# rgb_coord=[]
# for j in range(0,21):
#     rgb_coord.append(coord_lister(res_bar[j].loc[0][2]))
# rgb_coord.append(coord_lister(res_bar[21].loc[0][2][1]))
# for j in range(22,28):
#     rgb_coord.append(coord_lister(res_bar[j].loc[0][2]))


tx=[]
ty=[]
for i in range(0,28):
    tmpx=[]
    tmpy=[]
    for j in range(0,len(qaz[i])):
        tmpx.append(qaz[i][j][0])
        tmpy.append(qaz[i][j][1])
    tx.append(tmpx)
    ty.append(tmpy)

legend_properties = {'weight':'bold', 'size': 15}
# plt.figure(figsize=(18,25))
# for i in range (0,28):
#     plt.subplot(7,4,i+1)
#     plt.title(camera[i],fontsize=15, fontweight = 'bold')
#     plt.xlabel('x', fontweight = 'bold', fontsize=15)
#     plt.ylabel('y', fontweight = 'bold', fontsize=15)
#     plt.plot(rr[i,:],gg[i,:],label=('camera, {}'.format(area_camera[i])))
#     plt.plot(r_Bar,g_Bar,label=('OBS, {}'.format(area_OBS)))
#     plt.plot(rr[i,:],gg[i,:], linewidth = 3,
#               label=('share, {}'.format(*area_res_bar[i])))
#     plt.fill_between(rr[i,:],gg[i,:])
#     plt.xticks(size = 12, fontweight = 'bold')
#     plt.yticks(size = 12, fontweight = 'bold')
#     plt.legend(prop=legend_properties)
#     plt.tight_layout()
# plt.savefig('new_rgb2',dpi=300)
# plt.close()


plt.figure(figsize=(40,20))
for i in range (0,28):
    plt.subplot(4,7,i+1)
    plt.title(camera[i],fontsize=15, fontweight = 'bold')
    plt.xlabel('x', fontweight = 'bold', fontsize=15)
    plt.ylabel('y', fontweight = 'bold', fontsize=15)
    plt.plot(rr[i,:],gg[i,:],label=('camera, {}'.format(area_camera[i])))
    plt.plot(x_D65,y_D65,label=('OBS, {}'.format(area_OBS)))
    plt.plot(tx[i],ty[i], linewidth = 3,
              label=('share, {}'.format(*area_res[i])))
    plt.fill_between(tx[i],ty[i])
    plt.xticks(size = 15, fontweight = 'bold')
    plt.yticks(size = 15, fontweight = 'bold')
    plt.legend(prop=legend_properties)
    plt.tight_layout()
plt.savefig('new_share',dpi=300)
plt.close()

