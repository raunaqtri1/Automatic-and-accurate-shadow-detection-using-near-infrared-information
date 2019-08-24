from osgeo import gdal
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import cv2
from skimage import morphology
from mahotas.labeled import bwperim

alpha = 14.0
beta = 0.5
gamma = 1.5
tao = 10.0

def f(x):
	return 1/(1+np.exp(np.multiply((1-np.power(x,(1/gamma))-beta),-alpha)))

filepath = r"BGR.tif"
raster = gdal.Open(filepath)
rasterArray = raster.ReadAsArray()

filepath = r"NIR.tif"
raster2 = gdal.Open(filepath)
rasterArray2 = raster2.ReadAsArray()

img = Image.fromarray(np.transpose(rasterArray[:3], (1, 2, 0)),mode='RGB')
img.save("RGB_sample.jpeg")
img = Image.fromarray(rasterArray2,mode='L')
img.save("NIR_sample.jpeg")
rasterArray = rasterArray/255.0
rasterArray2 = rasterArray2/255.0
Lij = (rasterArray[0] + rasterArray[1] + rasterArray[2])/3.0
Dvis = f(Lij)
img = Image.fromarray(np.multiply(Dvis,255.0).astype(np.uint8),mode='L')
img.save("Dvis_sample.jpeg")
Lij= None
Dnir = f(rasterArray2)
img = Image.fromarray(np.multiply(Dnir,255.0).astype(np.uint8),mode='L')
img.save("Dnir_sample.jpeg")
D = np.multiply(Dvis,Dnir)
img = Image.fromarray(np.multiply(D,255.0).astype(np.uint8),mode='L')
img.save("D_sample.jpeg")
Dvis = None
Dnir = None
tRed = rasterArray[0]/rasterArray2
tGreen = rasterArray[1]/rasterArray2
tBlue = rasterArray[2]/rasterArray2
T = np.minimum(np.maximum.reduce([tRed,tGreen,tBlue]),tao)/tao
img = Image.fromarray(np.multiply(T,255.0).astype(np.uint8),mode='L')
img.save("T_sample.jpeg")
tRed = None
tGreen = None
tBlue = None
U = np.multiply((1-D),(1-T))
D = None
T = None
U = np.multiply(U,255.0).astype(np.uint8)
img = Image.fromarray(U,mode='L')
img.save("U_sample.jpeg")
theta = threshold_otsu(U)
median = cv2.medianBlur(np.multiply((U < theta).astype(np.uint8),255),3)
b = morphology.remove_small_objects(median.astype(np.bool), 15,connectivity=8)
b = morphology.remove_small_holes(b, 15,connectivity=8)
plt.axis('off')
plt.imsave('shadow_mask_sample.jpeg', b,cmap="gray", format="jpeg")
shadow_objects = bwperim(b,8)
plt.imsave('shadow_objects_sample.jpeg', shadow_objects,cmap="gray", format="jpeg")
