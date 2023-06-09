import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  


Data=np.array(pd.read_excel('.\Cameras 400-700.xlsx'))
OBS=np.array(pd.read_excel('.\OBS 400-700.xlsx'))
Name = np.array(pd.read_excel('.\Cameras_Name.xlsx'))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#--------------------------- CAMERA NAMES -------------------------

camera=['Canon 1DMarkIII','Canon 20D','Canon 300D',' Canon 40D',
        'Canon 500D','Canon 50D','Canon 5DMark II','Canon 600D',
        'Canon 60D','Hasselblad H2','Nikon D3X','Nikon D200',
        'Nikon D3','Nikon D300s','Nikon D40','Nikon D50',
        'Nikon D5100','Nikon D700','Nikon D80','Nikon D90',
        'Nokia N900','Olympus E-PL2','Pentax K-5','Pentax Q',
        'Point Grey Grasshopper 50S5C','Point Grey Grasshopper2 14S5C',
        'Phase One','SONY NEX-5N', 'Eye']
camera=np.array(camera)

camera2=['1 : Canon 1DMarkIII','2 : Canon 20D','3 : Canon 300D','4 : Canon 40D',
        '5 : Canon 500D','6 : Canon 50D','7 : Canon 5DMark II','8 : Canon 600D',
        '9 : Canon 60D','10 : Hasselblad H2','11 : Nikon D3X','12 : Nikon D200',
        '13 : Nikon D3','14 : Nikon D300s','15 : Nikon D40','16 : Nikon D50',
        '17 : Nikon D5100','18 : Nikon D700','19 : Nikon D80','20 : Nikon D90',
        '21 : Nokia N900','22 : Olympus E-PL2','23 : Pentax K-5','24 : Pentax Q',
        '25 : Point Grey Grasshopper 50S5C','26 : Point Grey Grasshopper2 14S5C',
        '27 : Phase One','28 : SONY NEX-5N', '29 : Eye']
camera2=np.array(camera2)

#---------------------------- CALCULATING ----------------------------

RGB_Bar=OBS[:,(6,7,8)]

R = np.array([i for i in Data[1:,1:85:3]])
G = np.array([i for i in Data[1:,2:85:3]])
B = np.array([i for i in Data[1:,3:85:3]])


landa = [i for i in range (400,710,10)]

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


RGB_rgb.append(np.hstack((RGB_Bar,RGB_Bar)))

# ----------------------------- CALCULATING SVD -------------------------------

uvw = []; u = []; v = []; w = []
for i in range (0,29):
    uvw.append(np.linalg.svd(RGB_rgb[i].astype(float)))
    u.append(uvw[i][0])
    v.append(uvw[i][1])
    w.append(uvw[i][2])

v = [np.diag(v[i]) for i in range (0,29)]

w = [w[i].T for i in range (0,29)]

#--------------------------- CALCULATING VORA ---------------------------

As=[]
for i in range (1,84,3):
    As.append(np.dot((Data[1:,i:i+3]).astype(float),np.linalg.pinv((Data[1:,i:i+3]).astype(float))))

RGBs = np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar))
cal = [np.dot(RGBs,As[i]) for i in range (0,28)]
Vora = [np.trace(cal[i]) for i in range (0,28)]
Vora.append(np.trace(np.dot(np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)),np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)))))

Vora = [Vora[i]/3 for i in range (0,29)]

#------------------------------ USING 3 VECTOR -------------------------

w_3 = []
for i in range (0,29):
    w_3.append(w[i].copy())
    
for i in range(0,29):
    w_3[i][:,3:] = 0

New_Data3 = [RGB_rgb[i].dot(w_3[i]) for i in range (0,29)]
Reproduct_Data3 = [(w_3[i].dot(New_Data3[i].T)).T for i in range (0,29)]

RMSE3 = []
for i in range (0,29):
    RMSE3.append(rmse(Reproduct_Data3[i], RGB_rgb[i]))
    

#------------------------------ USING 4 VECTOR -------------------------

w_4 = []
for i in range (0,29):
    w_4.append(w[i].copy())

for i in range(0,29):
    w_4[i][:,4:] = 0

New_Data4 = [RGB_rgb[i].dot(w_4[i]) for i in range (0,29)]
Reproduct_Data4 = [(w_4[i].dot(New_Data4[i].T)).T for i in range (0,29)]

RMSE4 = []
for i in range (0,29):
    RMSE4.append(rmse(Reproduct_Data4[i], RGB_rgb[i]))

#------------------------------ USING 5 VECTOR -------------------------

w_5 = []
for i in range (0,29):
    w_5.append(w[i].copy())
    
for i in range(0,29):
    w_5[i][:,5:] = 0

New_Data5 = [RGB_rgb[i].dot(w_5[i]) for i in range (0,29)]
Reproduct_Data5 = [(w_5[i].dot(New_Data5[i].T)).T for i in range (0,29)]

RMSE5 = []
for i in range (0,29):
    RMSE5.append(rmse(Reproduct_Data5[i], RGB_rgb[i]))

#------------------------------ USING 6 VECTOR -------------------------

w_6 = []
for i in range (0,29):
    w_6.append(w[i].copy())
    
for i in range(0,29):
    w_6[i][:,6:] = 0

New_Data6 = [RGB_rgb[i].dot(w_6[i]) for i in range (0,29)]
Reproduct_Data6 = [(w_6[i].dot(New_Data6[i].T)).T for i in range (0,29)]

RMSE6 = []
for i in range (0,29):
    RMSE6.append(rmse(Reproduct_Data6[i], RGB_rgb[i]))

#__________________________________________________________________________

RMSEs = np.vstack((RMSE3, RMSE4, RMSE5))
cam_rmse = [RMSEs[:, i] for i in range (0, 29)]

z = [np.hstack((cam_rmse[i].reshape(3,1), np.full((3,1),Vora[i]))) for i in range (0,29)]

RMSE3_del = np.delete(RMSE3, 24); RMSE3_del = np.delete(RMSE3_del, 27)
RMSE4_del = np.delete(RMSE4, 24); RMSE4_del = np.delete(RMSE4_del, 27)
RMSE5_del = np.delete(RMSE5, 23); RMSE5_del = np.delete(RMSE5_del, 27)
RMSE6_del = np.delete(RMSE6, 26); RMSE6_del = np.delete(RMSE6_del, 27)
Vora_del = np.delete(Vora, 24); Vora_del = np.delete(Vora_del, 27)

aspect_ratio = [(v[i][2, 2]/v[i][0, 0]).round(4) for i in range (0,29)]
aspect_ratio_del = np.delete(aspect_ratio, 24)
aspect_ratio_del = np.delete(aspect_ratio_del, 27)

corr_RMSE3_Vora = (np.corrcoef(RMSE3,Vora)[0,1]).round(4)
corr_RMSE4_Vora = (np.corrcoef(RMSE4,Vora)[0,1]).round(4)
corr_RMSE5_Vora = (np.corrcoef(RMSE5,Vora)[0,1]).round(4)
corr_RMSE6_Vora = (np.corrcoef(RMSE6,Vora)[0,1]).round(4)
corr_RMSE3_Vora_del = (np.corrcoef(RMSE3_del,Vora_del)[0,1]).round(4)
corr_RMSE4_Vora_del = (np.corrcoef(RMSE4_del,Vora_del)[0,1]).round(4)
corr_RMSE5_Vora_del = (np.corrcoef(RMSE5_del,Vora_del)[0,1]).round(4)
corr_RMSE6_Vora_del = (np.corrcoef(RMSE6_del,Vora_del)[0,1]).round(4)
corr_aspect_Vora = np.corrcoef(aspect_ratio, Vora)[0][1].round(4)

# _________________________________________________________________________

number_Dim = [3, 4, 5,]

m34 = []
for i in range (0,29):
    m34.append(z[i][1,0] - z[i][0,0])

m45 = []
for i in range (0,29):
    m45.append(z[i][2,0] - z[i][1,0])
    
m35 = []
for i in range (0,29):
    m35.append((z[i][2,0] - z[i][0,0])/2)
    
corr_m34_vora = np.corrcoef(m34, Vora)[0,1].round(4)
corr_m45_vora = np.corrcoef(m45, Vora)[0,1].round(4)
corr_m35_vora = np.corrcoef(m35, Vora)[0,1].round(4)


from sklearn.linear_model import LinearRegression

m_best = []
Y_pred = []
corr_mbest_vora = []
Y = np.array(number_Dim).reshape(-1, 1)
linear_regressor = LinearRegression()  
for i in range (0,29):
    m_best.append(np.array(z[i][:,0]).reshape(-1, 1))
    linear_regressor.fit(m_best[i], Y)  
    Y_pred.append(linear_regressor.predict(m_best[i]))

Y_pred_list = []
for i in range(0,29):
    j=0
    Y_pred_list.append([float(Y_pred[i][j]),float(Y_pred[i][j+1]),float(Y_pred[i][j+2])])
Y_list = [float(Y[0]),float(Y[1]),float(Y[2])]

Y_pred_list1 = []
Y_pred_list2 = []
Y_pred_list3 = []

for i in range (0,29):
    Y_pred_list1.append(Y_pred_list[i][0])
    Y_pred_list2.append(Y_pred_list[i][1])
    Y_pred_list3.append(Y_pred_list[i][2])

corr_best1 = (np.corrcoef(Y_pred_list1, Vora)[0,1]).round(4)
corr_best2 = (np.corrcoef(Y_pred_list2, Vora)[0,1]).round(4)
corr_best3 = (np.corrcoef(Y_pred_list3, Vora)[0,1]).round(4)

# plt.figure(figsize = (30,30))
# for i in range (0,29):
#     plt.subplot(6, 5, i+1)
#     plt.scatter(Y, m_best[i], label=(camera[i]))
#     plt.plot(Y_pred[i], m_best[i], color='red',) #label='R_square34 = {}'.format(R_square34))
#     plt.xlabel('Dim',fontsize=15, loc=('right'))
#     plt.ylabel('RMSE',fontsize=15, loc=('top'))
#     plt.xticks([3, 4, 5])
#     plt.yticks(np.arange(0,0.06,0.005))
#     plt.legend()
# plt.savefig('best_line vs Vora', dpi = 100)
# plt.close()

# plt.figure(figsize = (6, 6))
# plt.subplot(2, 2, 1)
# plt.scatter(m34, Vora, label='corr_m34_vora = {}'.format(corr_m34_vora))
# plt.subplot(2, 2, 2)
# plt.scatter(m45, Vora, label='corr_m45_vora = {}'.format(corr_m45_vora))
# plt.subplot(2, 2, 3)
# plt.scatter(m35, Vora, label='corr_m35_vora = {}'.format(corr_m35_vora))
# plt.subplot(2, 2, 4)
# plt.scatter(m_best, Vora, label='corr_mbest_vora = {}'.format(corr_mbest_vora))
# plt.xlabel('best_line',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# # plt.title('best_line vs Vora',fontsize=15,fontweight="bold")
# plt.legend()
# plt.savefig('best_line', dpi = 100)
# plt.close()


# plt.figure(figsize=(30,26))
# for i in range (0,29):
#     plt.subplot(6, 5, i+1)
#     plt.plot(number_Dim, z[i][:,1]/20, label='Vora', marker='o')
#     plt.plot(number_Dim, z[i][:,0], label=(camera[i]), marker='o')
#     plt.plot(number_Dim, m[i] * (np.array(number_Dim)) + b[i], '--' )
#     plt.xticks([3, 4, 5])
#     plt.yticks(np.arange(0,0.06,0.005))
#     plt.xlabel('Dim', fontweight='bold', fontsize=12)
#     plt.ylabel('RMSE', fontweight='bold', fontsize=12)
#     # ax.set_xticklabels(labels=['using 3 vector', 'using 4 vector', 'using 5 vector'])
#     # ax.annotate("Vora", xy=(4.9, 0.043),fontweight='bold' )
#     # fig.tight_layout()
#     plt.legend()
# plt.savefig('Dim vs RMSE', dpi=200)
# plt.close()

# RMSE34 = np.array(RMSE4) - np.array(RMSE3)
# RMSE45 = np.array(RMSE5) - np.array(RMSE4)
# RMSE35 = (np.array(RMSE5) - np.array(RMSE3))/2

# RMSEs2 = np.vstack((RMSE34, RMSE45, RMSE35))
# cam_rmse2 = [RMSEs2[:, i] for i in range (0, 29)]

# z2 = [np.hstack((cam_rmse2[i].reshape(3,1), np.full((3,1),Vora[i]))) for i in range (0,29)]

# plt.figure(figsize=(28,25))
# for i in range (0,29):
#     plt.subplot(6, 5, i+1)
#     plt.plot(number_Dim, np.abs(z2[i][:,0]), label=(camera[i]), marker='o')
#     plt.plot(number_Dim, z2[i][:,1]/20, label='Vora', marker='o')
#     plt.yticks(np.arange(0,0.05,0.005))
#     plt.xlabel('Dim', fontweight='bold', fontsize=12)
#     plt.ylabel('RMSE', fontweight='bold', fontsize=12)
#     plt.legend()
# plt.savefig('Dim vs RMSE_slope', dpi=100)
# plt.close()


# ------------------ PLOTTING ALL CAMERAS RMSEs AGAINST VORA -----------------

numbers = [i for i in range (1,30)]
plt.figure(figsize = (30, 25))
for i in range (0,29):
    plt.title('RMSEs VS. Vora', fontsize = 35)
    plt.plot(z[i][:,0],z[i][:,1], label=('{}'.format((numbers[i], camera[i]))))
    plt.scatter(z[i][:,0],z[i][:,1])
    plt.xlabel('RMSEs', fontsize = 35, loc=('right'))
    plt.ylabel('Vora', fontsize = 35, loc=('top'))
    plt.xlim(0, np.max(RMSEs))
    plt.ylim(0.69,1)
    plt.xticks(size = 35)
    plt.yticks(size = 35)
    plt.legend(prop={'size': 23})
    plt.annotate(numbers[i], xy=(RMSEs[0,i],z[i][0,1]), fontsize = 8)
    plt.tight_layout()
plt.savefig('camm2', dpi = 300)
plt.close()






# -----------------------------------------------------------------------------

# fig, ax = plt.subplots(1, 2, figsize=(36, 15))
# ax[0].scatter(aspect_ratio_del, Vora_del)
# ax[0].set_title('aspect_ratio vs vora', fontweight='bold', fontsize=15)
# ax[0].set_xlabel('aspect_ratio', fontweight = 'bold', fontsize=13)
# ax[0].set_ylabel('Vora', fontweight = 'bold', fontsize=13)

# ax[1].scatter(aspect_ratio_del, RMSE3_del)
# ax[1].set_title('aspect_ratio vs RMSE3', fontweight = 'bold', fontsize=15)
# ax[1].set_xlabel('aspect_ratio', fontweight = 'bold', fontsize=13)
# ax[1].set_ylabel('RMSE3', fontweight = 'bold', fontsize=13)
# # for i in range (0,27):
# #     ax[0].annotate(camera[i], xy = (aspect_ratio_del[i], Vora_del[i]))
# #     ax[1].annotate(camera[i], xy = (aspect_ratio_del[i], RMSE3_del[i]))
# fig.tight_layout
# plt.savefig('Aspect_ratio vs RMSE and Vora', dpi = 300)
# plt.close()

# ____________________________________________________________________________
