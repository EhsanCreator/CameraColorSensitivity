import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score  


Data=np.array(pd.read_excel('.\Cameras 400-700.xlsx'))
OBS=np.array(pd.read_excel('.\OBS 400-700.xlsx'))

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
        'Phase One','SONY NEX-5N', 'OBS']
camera=np.array(camera)

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

RGBs = np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar))

RGB_rgb.append(np.hstack((RGB_Bar,RGB_Bar)))

uvw = []; u = []; v = []; w = []
for i in range (0,29):
    uvw.append(np.linalg.svd(RGB_rgb[i].astype(float)))
    u.append(uvw[i][0])
    v.append((uvw[i][1]).round(4))
    w.append(uvw[i][2])

v = [np.diag(v[i]) for i in range (0,29)]
v = [np.vstack((v[i],np.zeros((25,6)))) for i in range (0,29)]

w = [w[i].T for i in range (0,29)]

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

# #------------------------------ USING 5 VECTOR -------------------------

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

#-------------------------- PLOTING 3_4_5 VECTOR -------------------------

line_labels = ["Rep5_R", "Rep5_G", "Rep5_B", "R", 'G', 'B']

plt.figure(figsize=(40,20))
for i in range (0,28):
    plt.subplot(4,7,i+1)
    plt.title(camera[i],fontsize=17, fontweight='bold')
    plt.plot(landa,Reproduct_Data3[i][:,3],label=('Rep3_R'),color='r')
    plt.plot(landa,Reproduct_Data3[i][:,4],label=('Rep3_G'),color='g')
    plt.plot(landa,Reproduct_Data3[i][:,5],label=('Rep3_B'),color='b')
    plt.xlabel('Lambda (nm)',loc='right',fontsize=15, fontweight='bold')
    plt.plot(landa,R[:,i],label=('R'),color='crimson')
    plt.plot(landa,G[:,i],label=('G'),color='lime')
    plt.plot(landa,B[:,i],label=('B'),color='aqua')
    # plt.legend()
    plt.xticks(size = 15, fontweight = 'bold')
    plt.yticks(size = 15, fontweight = 'bold')
    plt.tight_layout()
    # plt.figlegend(labels=line_labels, loc="upper right", 
    #               borderaxespad=0.1, 
    #               title="Legend Title")
plt.savefig('3vec_new',dpi=300)
plt.close()



# ________________________ Bar PLOTING RMSEs _________________________

# l = [i for i in range (0,29)]
# l = np.array(l)

# RMSE_df = pd.DataFrame({'RMSE3' : RMSE3, 'RMSE4' : RMSE4, 'RMSE5' : RMSE5},
#                        index = [camera])


# ax = RMSE_df.plot.bar(figsize=(20,20), fontsize=10)
# plt.title('RMSEs for each camera')
# plt.xlabel('Camera')
# plt.ylabel('RMSE')
# plt.savefig('RMSE3456', dpi=100)

#---------------------------------------------------------------------

# As=[]
# for i in range (1,84,3):
#     As.append(np.dot((Data[1:,i:i+3]).astype(float),np.linalg.pinv((Data[1:,i:i+3]).astype(float))))

# cal = [np.dot(RGBs,As[i]) for i in range (0,28)]
# Vora = [np.trace(cal[i]) for i in range (0,28)]
# Vora.append(np.trace(np.dot(np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)),np.dot(RGB_Bar,np.linalg.pinv(RGB_Bar)))))

# Vora = [Vora[i]/3 for i in range (0,29)]
# Vora_df = pd.DataFrame({'Vora' : Vora,},index = [camera])



# RMSE34 = list(np.array(RMSE3) - np.array(RMSE4))
# RMSE45 = list(np.array(RMSE4) - np.array(RMSE5))

# RMSE34_df = pd.DataFrame({'RMSE34' : RMSE34,},index = [camera])

# RMSE45_df = pd.DataFrame({'RMSE45' : RMSE45},index = [camera])

# corr_RMSE34_Vora = (np.corrcoef(RMSE34,Vora)[0,1]).round(4)
# corr_RMSE45_Vora = (np.corrcoef(RMSE45,Vora)[0,1]).round(4)
# RMSE34 = [RMSE34[i]*100 for i in range (0,29)]
# RMSE45 = [RMSE45[i]*100 for i in range (0,29)]


# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.scatter(RMSE34,Vora, label=('corr= {}'.format(corr_RMSE34_Vora)))
# plt.title('RMSE34 VS. Vora',fontweight="bold",fontsize=15)
# plt.xlabel('RMSE34',fontsize=13)
# plt.ylabel('Vora',fontsize=13)
# plt.annotate('Point Grey Grasshopper 50S5C', xy= (1.4,0.69), fontsize=7)
# plt.annotate('OBS', xy= (0.03,1), fontsize=7)
# plt.legend()
# plt.subplot(1,2,2)
# plt.scatter(RMSE45,Vora, label=('corr= {}'.format(corr_RMSE45_Vora)))
# plt.title('RMSE45 VS. Vora',fontweight="bold",fontsize=15)
# plt.xlabel('RMSE45',fontsize=13)
# plt.ylabel('Vora',fontsize=13)
# plt.annotate('Point Grey Grasshopper 50S5C', xy= (1.32,0.69), fontsize=7)
# plt.annotate('OBS', xy= (0.03,1), fontsize=7)
# plt.legend()
# plt.tight_layout()
# plt.savefig("RMSE_34_45_Vora_old",dpi=200)
# plt.close()

# RMSE34.pop(24); RMSE34.pop(27)
# RMSE45.pop(24); RMSE45.pop(27)
# Vora.pop(24); Vora.pop(27)
# corr_RMSE34_Vora_new = (np.corrcoef(RMSE34,Vora)[0,1]).round(4)
# corr_RMSE45_Vora_new = (np.corrcoef(RMSE45,Vora)[0,1]).round(4)


# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.scatter(RMSE34[2:29],Vora[2:29], label=('corr= {}'.format(corr_RMSE34_Vora_new)))
# plt.title('RMSE34 VS. Vora',fontweight="bold",fontsize=15)
# plt.xlabel('RMSE34',fontsize=13)
# plt.ylabel('Vora',fontsize=13)
# plt.legend()
# plt.subplot(1,2,2)
# plt.scatter(RMSE45[2:29],Vora[2:29], label=('corr= {}'.format(corr_RMSE45_Vora_new)))
# plt.title('RMSE45 VS. Vora',fontweight="bold",fontsize=15)
# plt.xlabel('RMSE45',fontsize=13)
# plt.ylabel('Vora',fontsize=13)
# plt.legend()
# plt.tight_layout()
# plt.savefig("RMSE_34_45_Vora_New",dpi=200)
# plt.close()

# # _________________________ Linear Regression _____________________________

# from sklearn.linear_model import LinearRegression

# X_34 = np.array(RMSE34).reshape(-1, 1)
# X_45 = np.array(RMSE45).reshape(-1, 1)
# Y = np.array(Vora).reshape(-1, 1)
# linear_regressor = LinearRegression()  
# linear_regressor.fit(X_34, Y)  
# linear_regressor.fit(X_45, Y)  
# Y_pred34 = linear_regressor.predict(X_34)
# Y_pred45 = linear_regressor.predict(X_45)  


# R_square34 = (r2_score(Y, Y_pred34)).round(4)
# R_square45 = (r2_score(Y, Y_pred45)).round(4)

# corr_RMSE34_Vora = (np.corrcoef(RMSE34,Vora)[0,1]).round(4)
# corr_RMSE45_Vora = (np.corrcoef(RMSE45,Vora)[0,1]).round(4)

# plt.figure(figsize=(12,4))
# plt.subplot(1,2,1)
# plt.scatter(X_34, Y, label='corr34 = {}'.format(corr_RMSE34_Vora))
# plt.plot(X_34, Y_pred34, color='red', label='R_square34 = {}'.format(R_square34))
# plt.xlabel('RMSE34',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# plt.title('RMSE34 VS Vora',fontsize=15,fontweight="bold")
# plt.legend()
# plt.subplot(1,2,2)
# plt.scatter(X_45, Y, label='corr45 = {}'.format(corr_RMSE45_Vora))
# plt.plot(X_45, Y_pred45, color='red', label='R_square45 = {}'.format(R_square45))
# plt.xlabel('RMSE45',fontsize=15)
# plt.ylabel('Vora',fontsize=15)
# plt.title('RMSE45 VS Vora',fontsize=15,fontweight="bold")
# plt.legend()
# plt.savefig('Corr_R2',dpi=300)
# plt.close()



 

