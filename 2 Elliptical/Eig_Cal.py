import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

Data = np.array(pd.read_excel('.\Cameras 400-700.xlsx'))
OBS = np.array(pd.read_excel('.\OBS 400-700.xlsx'))

camera=['Canon 1DMarkIII','Canon 20D','Canon 300D',' Canon 40D',
        'Canon 500D','Canon 50D','Canon 5DMark II','Canon 600D',
        'Canon 60D','Hasselblad H2','Nikon D3X','Nikon D200',
        'Nikon D3','Nikon D300s','Nikon D40','Nikon D50',
        'Nikon D5100','Nikon D700','Nikon D80','Nikon D90',
        'Nokia N900','Olympus E-PL2','Pentax K-5','Pentax Q',
        'Point Grey Grasshopper 50S5C','Point Grey Grasshopper2 14S5C',
        'Phase One','SONY NEX-5N','OBS']
camera=np.array(camera)

#-------------------- CALCULATING COVARIANCE MATRIX ---------------------

RGB_Bar = OBS[:,(6,7,8)]

RGB_rgb = []
for i in range (1, 84, 3):
    RGB_rgb.append(np.hstack((RGB_Bar, Data[1:,i:i+3])))

RGB_rgb.append(np.hstack((RGB_Bar, RGB_Bar)))

cov_mat = []
for i in range (0, 29):
    cov_mat.append(np.cov(RGB_rgb[i][:, 0:3].astype(float), RGB_rgb[i][:, 3:].astype(float), rowvar=False).round(4))

cor_mat = []
for i in range (0,29):
    cor_mat.append(np.corrcoef(RGB_rgb[i][:, 0:3].astype(float), RGB_rgb[i][:, 3:].astype(float), rowvar=False).round(4))


#-----------------------------------------------------------------------

def confidence_ellipse(x, y, ax, n_std=3, **kwargs):

    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    
    cov = np.cov(x.T, y.T)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1,1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0,0), width=ell_radius_x * 2, height=ell_radius_y * 2, **kwargs)

    # calculating the stdandarddeviation of x from  the squareroot of the variance
    # np.sqrt(cov[0, 0])
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    
    # calculating the stdandarddeviation of y from  the squareroot of the variance
    # np.sqrt(cov[1, 1])
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    
    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)
        
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    
        
    return pearson        
    


fig, ax0 = plt.subplots()
(confidence_ellipse(RGB_Bar, RGB_rgb[0][:,3:].astype('float'), ax0, facecolor='w', edgecolor='b'))
plt.xlim(-1,1)
plt.ylim(-1,1)
(confidence_ellipse(RGB_Bar, RGB_rgb[1][:,3:].astype('float'), ax0, facecolor='w', edgecolor='b'))
(confidence_ellipse(RGB_Bar, RGB_rgb[2][:,3:].astype('float'), ax0, facecolor='w', edgecolor='b'))
(confidence_ellipse(RGB_Bar, RGB_rgb[3][:,3:].astype('float'), ax0, facecolor='w', edgecolor='b'))

ax1 = plt.subplot(131)
(confidence_ellipse(RGB_Bar, RGB_rgb[0][:,3:].astype('float'), ax1, facecolor='w', edgecolor='b'))
ax1.set_xlim([-1, 1])
ax1.set_ylim([-1, 1])

plt.figure(figsize=(30,30))
for i in range (0,29):
    ax = plt.subplot(6, 5, i+1)
    ax.plot((confidence_ellipse(RGB_rgb[i][:,3:].astype('float'), RGB_Bar, ax, edgecolor='b', facecolor='w')))
    ax.scatter(RGB_rgb[i][:,3:].astype('float'), RGB_Bar)
    ax.set_xlabel('x', loc='right', fontweight="bold")
    ax.set_ylabel('y', loc='top', fontweight="bold")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.title(camera[i],fontsize=14,fontweight="bold")
    fig.tight_layout()
plt.savefig('Elip 2D',dpi=300)
plt.close()

# area = []
# for i in range(0,28):
#     area_res.append((.area).round(4))


# ax = plt.subplot(1,1,1)
# ax.plot((confidence_ellipse(RGB_rgb[0][:,3:].astype('float'), RGB_Bar, ax, edgecolor='b', facecolor='w', )))
# # ax.scatter(RGB_rgb[0][:,3:].astype('float'), RGB_Bar)
# ax.scatter(RGB_rgb[0][:,3:].astype('float'), RGB_Bar)







