from PIL import Image, ImageColor
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import h5py
import sys
np.set_printoptions(threshold=sys.maxsize)
index = []

# # #########################Check which runs failed############################
# # #deformed images
# # # for i in range(1,2001):
# # # 	if not os.path.exists('deformed_images/deformed'+str(i)+'.png'):
# # # 		index.append(i)
# # # print(index)
# # # #data
# # # for i in range(1,2001):
# # # 	if not os.path.exists('data/data'+str(i)+'.npy'):
# # # 		index.append(i)
# # # print(index)
# # ############################################################################

##check which deformed images exist
for i in range(0,len(os.listdir('deformed_images'))):
	if os.listdir('deformed_images')[i][8:11].isdigit():
		index.append(int(os.listdir('deformed_images')[i][8:11]))
	elif os.listdir('deformed_images')[i][8:10].isdigit():
		index.append(int(os.listdir('deformed_images')[i][8:10]))
	else:
		index.append(int(os.listdir('deformed_images')[i][8:9]))
#crop images to desired size and put into array
#put data into array
num_images = len(index)
size = 256
resize = 32
# print(num_images)
og_images = np.zeros((num_images,resize,resize,3),dtype=int)
# deformed_images = np.zeros((num_images,size,size,3))
data = np.zeros((num_images,5,145))
E = np.zeros((num_images))
for i,idx in enumerate(index):
	#original images
	ogimage = Image.open('og_images/og'+str(idx)+'.png')
	ogcropped = ogimage.crop((181, 55, 540, 410))
	ogcropped = ogcropped.resize((size,size),Image.ANTIALIAS)
	ogcropped = ogcropped.convert('RGB')
	ogcropped = np.array(ogcropped)
	for p in range(0,256):
		for q in range(0,256):
			if np.all(ogcropped[p,q,:] == 254):
				ogcropped[p,q,:] = np.array([0,0,0])
	ogcropped = Image.fromarray(ogcropped)
	ogcropped = ogcropped.resize((32,32),Image.ANTIALIAS)
	og_images[i-1,:,:,:] = np.array(ogcropped)

	# #deformed images
	# deformedimage = Image.open('deformed_images/deformed'+str(idx)+'.png')
	# deformedcropped = deformedimage.crop((181, 50, 546, 415))
	# deformedcropped = deformedcropped.resize((size,size),Image.ANTIALIAS)
	# deformedcropped = deformedcropped.convert('RGB')
	# deformed_images[i-1,:,:,:] = np.array(deformedcropped)

	#data
	data_int = np.load('data/data'+str(idx)+'.npy')
	data_int = np.load('data/data'+str(idx)+'.npy')[:,:,0:145].reshape((5,145))
	data_int[0:4,:] = data_int[0:4,:]
	# print(data_int.shape)
	data[i,:,:] = data_int/1e6

	#youngs modulus
	E[i] = ((data_int[1,10]-data_int[1,9])/((data_int[4,10]-data_int[4,9])*0.006))/1e9

# #make hdf5 dataset
d = h5py.File('porositydataset_resized.hdf5','a')
d.create_dataset('Original Images', data = og_images)
# d.create_dataset('Deformed Images'.encode('utf-8'), data = deformed_images)
d.create_dataset('Data'.encode('utf-8'), data = data)
d.create_dataset('Youngs Modulus'.encode('utf-8'), data = E)

#####make plot
# data_int = np.load('data/data'+str(1)+'.npy').reshape((5,145))
# print(data_int[4,:])
# strain = data_int[4,:]*0.006
# E = ((data_int[1,10]-data_int[1,9])/((data_int[4,10]-data_int[4,9])*0.006))/1e9
# print(E)
# plt.plot(strain,data_int[0,:]/1e6,label='von Mises Stress')
# plt.plot(strain,data_int[1,:]/1e6,label='Stress in X Direction')
# plt.plot(strain,data_int[2,:]/1e6,label='Stress in Y Direction')
# plt.plot(strain,data_int[3,:]/1e6,label='Pressure Stress')
# plt.legend()
# plt.xlabel('Strain')
# plt.ylabel('Stress (MPa)')
# plt.text(0.0005,50,'Young\'s Modulus:\n 181.16 GPa') 
# plt.grid()
# plt.show()
# data_int = np.load('data/data'+str(idx)+'.npy')[:,:,0:145].reshape((5,145))
# data_int[0:4,:] = data_int[0:4,:]
# # print(data_int.shape)
# data[i,:,:] = data_int/1e6
# print(ImageColor.getcolor('red','H'))