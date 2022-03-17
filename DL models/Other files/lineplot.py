#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 23:58:35 2021

@author: fistlab
"""

## import all the necessary packages
from skimage.measure import profile_line
import scipy.io
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as font_manager



## Read all the mat files
u=scipy.io.loadmat('/media/fistlab/raj/line/unet.mat')
fdu=scipy.io.loadmat('/media/fistlab/raj/line/FDUnet.mat')
y=scipy.io.loadmat('/media/fistlab/raj/line/Ynet.mat')
fdy=scipy.io.loadmat('/media/fistlab/raj/line/FDYnet.mat')
res=scipy.io.loadmat('/media/fistlab/raj/line/resnet.mat')
gan=scipy.io.loadmat('/media/fistlab/raj/line/gan.mat')


# u=scipy.io.loadmat('/media/fistlab/DATA/Line profile/Unet.mat')
# fdu=scipy.io.loadmat('/media/fistlab/DATA/Line profile/FD Unet.mat')
# y=scipy.io.loadmat('/media/fistlab/DATA/Line profile/Ynet.mat')
# fdy=scipy.io.loadmat('/media/fistlab/DATA/Line profile/FDYnet.mat')
# res=scipy.io.loadmat('/media/fistlab/DATA/Line profile/Resnet.mat')
# gan=scipy.io.loadmat('/media/fistlab/DATA/Line profile/GAN.mat')



### first line profile
# gt = u['GT']
# i = u['inp']
# p1 = u['P']
# p2 = fdu['P']
# p3 = y['P']
# p4 = fdy['P']
# p5 = res['P']
# p6 = gan['P']

# ############
# gt = u['o1']
# i = u['i1']
# p1 = u['P1']
# p2 = fdu['P1']
# p3 = y['P1']
# p4 = fdy['P1']
# p5 = res['P1']
# p6 = gan['P1']

# ############
# gt = u['o2']
# i = u['i2']
# p1 = u['P2']
# p2 = fdu['P2']
# p3 = y['P2']
# p4 = fdy['P2']
# p5 = res['P2']
# p6 = gan['P2']
##############
gt = u['o4']
# i = u['i3']
# p1 = u['P3']
# p2 = fdu['P3']
# p3 = y['P3']
# p4 = fdy['P3']
# p5 = res['P3']
# p6 = gan['P3']
# # #################
# gt = u['o4']
# i = u['i4']
# p1 = u['P4']
# p2 = fdu['P4']
# p3 = y['P4']
# p4 = fdy['P4']
# p5 = res['P4']
# p6 = gan['P4']
#####################
# gt = u['o5']
i = u['i5']
p1 = u['P5']
p2 = fdu['P5']
p3 = y['P5']
p4 = fdy['P5']
p5 = res['P5']
p6 = gan['P5']
# # ###############
# # im = plt.imshow(gt, cmap = 'gray')
# # plt.colorbar(fontsize=16,fontweight="bold")
# cbar = plt.colorbar(im)
# tick_font_size = 10
# cbar.ax.tick_params(labelsize=tick_font_size)

# # plt.clim(-0.02, 0.04);


# plt.imshow(p6,cmap='gray')
# cropped = p6[220:250, 50:80]
# cropped = gt[185:250, 5:210]
# plt.imshow(cropped, cmap='gray')
# plt.imsave("/media/fistlab/DATA/Line profile/gt30_cropped.png", cropped.squeeze(), cmap='gray')
# plt.axis('off')

gt  = np.flip(gt,0)
# # plt.imshow(p3)
i  = np.flip(i,0)
p1  = np.flip(p1,0)
p2  = np.flip(p2,0)
p3  = np.flip(p3,0)
p4  = np.flip(p4,0)
p5  = np.flip(p5,0)
p6  = np.flip(p6,0)



# start1=(120,40)
# end1=(120,55)
# start=(10,50)
# end=(10,60)



start=(10,150) #Blue line (y,x)
end=(10,160)
start1=(100,170) # orange line
end1=(100,185)

# # # plt.imshow(p6, cmap = 'gray')
## line profile 1 orange line
profile = profile_line(gt,start1,end1, linewidth=2)
profile1 = profile_line(i,start1,end1, linewidth=2)
profile2 = profile_line(p1,start1,end1, linewidth=2)
profile3 = profile_line(p2,start1,end1, linewidth=2)
profile4 = profile_line(p3,start1,end1, linewidth=2)
profile5 = profile_line(p4,start1,end1, linewidth=2)
profile6 = profile_line(p5,start1,end1, linewidth=2)
profile7 = profile_line(p6,start1,end1, linewidth=2)


# # ###line profile 2 blue line

# profile = profile_line(gt,start,end, linewidth=2)
# profile1 = profile_line(i,start,end, linewidth=2)
# profile2 = profile_line(p1,start,end, linewidth=2)
# profile3 = profile_line(p2,start,end, linewidth=2)
# profile4 = profile_line(p3,start,end, linewidth=2)
# profile5 = profile_line(p4,start,end, linewidth=2)
# profile6 = profile_line(p5,start,end, linewidth=2)
# profile7 = profile_line(p6,start,end, linewidth=2)




plt.figure(figsize=(12., 6.))
plt.imshow(gt, cmap='gray',origin='lower',alpha=1)
plt.plot([start[1],end[1]],[start[0],end[0]],'tab:blue',lw=3)
plt.imshow(gt, cmap='gray',origin='lower',alpha=1)
plt.plot([start1[1],end1[1]],[start1[0],end1[0]],'tab:orange',lw=3)
plt.axis('off')
plt.figure(figsize=(12., 6.))
plt.imshow(i, cmap='gray',origin='lower',alpha=1)
plt.plot([start[1],end[1]],[start[0],end[0]],'tab:blue',lw=3)
plt.imshow(i, cmap='gray',origin='lower',alpha=1)
plt.plot([start1[1],end1[1]],[start1[0],end1[0]],'tab:orange',lw=3)
# plt.axis('off')



font2 = {'family':'Comic Sans MS','color':'k','size':25}
font1 = {'family':'Comic Sans MS','color':'k','size':16}
font = font_manager.FontProperties(family='Comic Sans MS',
                                    weight='bold',
                                    style='normal', size=16)


plt.figure(figsize=(12., 6.))
plt.plot(profile,color='k', marker='.', linestyle='-',linewidth=3, markersize=16, label = 'Ground Truth')
plt.plot(profile1,color='red', marker='o', linestyle='--',linewidth=3, markersize=12, label ='TR')
plt.plot(profile2,color='b', marker='v', linestyle='-.',linewidth=3, markersize=12, label = 'U-Net' )
plt.plot(profile3,color='darkorange', marker='8', linestyle=':',linewidth=3, markersize=12, label = 'FD U-Net')
plt.plot(profile4,color='darkviolet', marker='p', linestyle='-',linewidth=3, markersize=12, label = 'Y-Net' )
plt.plot(profile5,color='tab:brown', marker='h', linestyle='--',linewidth=3, markersize=12,label = 'FD Y-Net' )
plt.plot(profile6,color='pink', marker='+', linestyle='-.',linewidth=3, markersize=20, label = 'Res U-Net')
plt.plot(profile7,color='g', marker='d', linestyle=':',linewidth=3, markersize=12, label = 'GAN')
# plt.axis('off')
# plt.ylim([-0.01, 0.05])
# plt.xticks(fontsize=16,fontweight="bold")
# plt.yticks(fontsize=16,fontweight="bold")
plt.xticks(np.linspace(0, 15, 3, dtype=int), fontsize=12,fontweight="bold")
plt.yticks(np.linspace(-0.01, 0.04, 3), fontsize=12,fontweight="bold")
# plt.xlabel('Distance (pixels)',fontdict = font1,fontstyle='normal', fontweight="bold")
# plt.ylabel('μₐ(cm⁻¹)',fontdict = font1,fontweight="bold",fontstyle='normal')
# plt.title('Line profile 1(40 dB,700nm)',fontdict = font1,fontweight="bold")
# plt.legend(prop=font, loc='best')






# import numpy as np
# import matplotlib.pyplot as plt
# import glob
# import scipy.io
# import matplotlib.font_manager as font_manager


# # a1 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_1.mat')
# # a2 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_2.mat')
# # a3 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_3.mat')
# # a4 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_4.mat')
# # a5 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_4.mat')
# # a6 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_4.mat')
# # a7 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_4.mat')
# # a8 = scipy.io.loadmat(r'C:\\Users\\arumugaraj\\Desktop\New folder (5)\data\\X_4.mat')

# A1 =gt.squeeze()
# A2 = i.squeeze()
# A3 = p1.squeeze()
# A4 = p2.squeeze()
# A5 = p3.squeeze()
# A6 = p4.squeeze()
# A7 = p5.squeeze()
# A8 = p6.squeeze()


# font1 = {'family':'DejaVu Sans','color':'k','size':16}
# import matplotlib.pyplot as plt
# import numpy as np

# def img_is_color(img):

#     if len(img.shape) == 3:
#         # Check the color channels to see if they're all the same.
#         c1, c2, c3 = img[:, : , 0], img[:, :, 1], img[:, :, 2]
#         if (c1 == c2).all() and (c2 == c3).all():
#             return True

#     return False

# def show_image_list(list_images, list_titles=None, list_cmaps=None, grid=None, num_cols=2, figsize=(50, 50), title_fontsize=30):
#     '''
#     Shows a grid of images, where each image is a Numpy array. The images can be either
#     RGB or grayscale.

#     Parameters:
#     ----------
#     images: list
#         List of the images to be displayed.
#     list_titles: list or None
#         Optional list of titles to be shown for each image.
#     list_cmaps: list or None
#         Optional list of cmap values for each image. If None, then cmap will be
#         automatically inferred.
#     grid: boolean
#         If True, show a grid over each image
#     num_cols: int
#         Number of columns to show.
#     figsize: tuple of width, height
#         Value to be passed to pyplot.figure()
#     title_fontsize: int
#         Value to be passed to set_title().
#     '''

#     assert isinstance(list_images, list)
#     assert len(list_images) > 0
#     assert isinstance(list_images[0], np.ndarray)

#     if list_titles is not None:
#         assert isinstance(list_titles, list)
#         assert len(list_images) == len(list_titles), '%d imgs != %d titles' % (len(list_images), len(list_titles))

#     if list_cmaps is not None:
#         assert isinstance(list_cmaps, list)
#         assert len(list_images) == len(list_cmaps), '%d imgs != %d cmaps' % (len(list_images), len(list_cmaps))

#     num_images  = len(list_images)
#     num_cols    = min(num_images, num_cols)
#     num_rows    = int(num_images / num_cols) + (1 if num_images % num_cols != 0 else 0)

#     # Create a grid of subplots.
#     fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
#     # Create list of axes for easy iteration.
#     if isinstance(axes, np.ndarray):
#         list_axes = list(axes.flat)
#     else:
#         list_axes = [axes]

#     for i in range(num_images):

#         img    = list_images[i]
#         title  = list_titles[i] if list_titles is not None else 'Image %d' % (i)
#         cmap   = list_cmaps[i] if list_cmaps is not None else (None if img_is_color(img) else 'gray')
        
#         list_axes[i].imshow(img, cmap=cmap)
#         list_axes[i].set_title(title, fontdict = font1,fontweight="bold")#fontsize=title_fontsize) 
#         list_axes[i].grid(grid)
#         list_axes[i].axis('off')

#     for i in range(num_images, len(list_axes)):
#         list_axes[i].set_visible(False)

#     # fig.tight_layout()
#     plt.subplots_adjust(left=0.125,
#                     bottom=0.1, 
#                     right=0.9, 
#                     top=0.55, 
#                     wspace=0.02, 
#                     hspace=0)

#     _ = plt.show()
    




# list_images = [A1,A2,A3,A4,A5,A6,A7,A8]

# show_image_list(list_images=[A1,A2,A3,A4,A5,A6,A7,A8], 
#                 list_titles=['(a) Reference', '(b) TR', '(c) U - Net', '(d) FD U - Net', '(e) Y - Net', '(f) FD Y - Net', '(g) Deep Resnet', '(h) GAN'],
#                 num_cols=4,
#                 figsize=(12, 12),
#                 grid=False,
#                 title_fontsize=30)

