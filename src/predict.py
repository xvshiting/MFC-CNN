import numpy
from PIL import Image
import binascii
import os
import sys
import random   
import numpy as np
import CNN_Model
def getMatrix_From_Bin(content,width):
    hexst=binascii.hexlify(content)
#     print(hexst[100:1200])
#     print(type(hexst))
    fh=numpy.array([int(hexst[i:i+2],16)for i in range(0,len(hexst),2)])
#     print(fh.shape)
    rn=len(fh)/width
    fh=numpy.reshape(fh[:int(rn)*width],(-1,width))
#     print(fh.shape)
    fh=numpy.uint8(fh)
    return fh
def  create_image(content):
	 im=Image.fromarray(getMatrix_From_Bin(content,1024))
	 im=im.resize((32,32),Image.ANTIALIAS)
	 im.show()
	 return im
def  create_data(content):
	image_data=np.array(create_image(content))
	return np.multiply(np.reshape(image_data,(1,32*32)),1.0 / 255.0)
def  predict(content):
	# print(content)
	str=CNN_Model.predict(data=create_data(content))
	print(str)
	return str
def ppp(file_path):
	with open(file_path,'rb')as f:
		content=f.read()
		return predict(content)
def pppp(file_path):
	with open(file_path,'rb')as f:
		im=np.array(Image.open(f))
		im=np.reshape(im,(1,32*32))
		str=CNN_Model.predict(data=create_data(im))
		print(str)
		return str

if __name__ == '__main__':
	file_path=sys.argv[1]
	with open(file_path,'rb')as f:
		content=f.read()
		predict(content)