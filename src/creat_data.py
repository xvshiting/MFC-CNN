import numpy as np
import os
import pickle as pk
from PIL import Image
import matplotlib.pylab as plt
import sys
def get_data_path(file_path):
	return os.sep.join(str.split(file_path,os.sep)[:-1])+os.sep+"data"
def process(dir_path):
	root=dir_path
	save_path=get_data_path(root)
	try:
		os.listdir(save_path)
	except :
		os.mkdir(save_path)
		print("creat "+ save_path)
	dirs=os.listdir(root)
	print(save_path)
#记录恶意软件家族与类别映射
	cl_map={}
	i=1
#恶意家族总数
	cl_num=0
	for file in dirs:
		if file!='.DS_Store':
			cl_map[file]=i
			i=i+1
			cl_num=i-1
	try:
		pkl_file = open(os.path.join(save_path,'data.pkl'), 'rb')
		cl_map_1=pk.load(pkl_file)
		pkl_file.close()
		output = open(os.path.join(save_path,'data.pkl'), 'wb')
		pk.dump(obj=[cl_map,cl_num],file=output)
		output.close()
	except :
		output = open(os.path.join(save_path,'data.pkl'), 'wb')
		pk.dump(obj=[cl_map,cl_num],file=output)
		output.close()
	print(cl_num,cl_map)

	image_data=[]
	image_label=[]
	for file in dirs:
		if file!='.DS_Store':
			im_files=os.listdir(os.path.join(root,file))
			for im_file in im_files:
				if im_file!='.DS_Store':
					im=np.array(Image.open(os.path.join(root,file,im_file)))
#                 	im=plt.imread(os.path.join(root,file,im_file))
					print(im.shape)
#                 	plt.imshow(im)
#                 	plt.show()
					im=np.reshape(im,(1,32*32))
					image_data.append(im)
					image_label.append(cl_map[file])

#One-hot 
	l=np.zeros((len(image_data),cl_num)) 
	label_offset=np.arange(len(image_data))*cl_num
	l.flat[label_offset+image_label-1]=1
	image_label=l
	image_data_out=open(os.path.join(save_path,'image_data.pkl'),'wb')
	lable_data_out=open(os.path.join(save_path,'image_label_data.pkl'),'wb')
	image_data=np.array(image_data)
	image_data=np.reshape(image_data,(image_data.shape[0],image_data.shape[2]))
	image_label=np.array(image_label)
	#shuffle data
	perm = np.arange(image_data.shape[0])
	np.random.shuffle(perm)
	image_data = image_data[perm]
	image_label=image_label[perm]
	pk.dump(image_data,image_data_out)
	pk.dump(image_label,lable_data_out)

def  creat_data():
	process("./image")

if __name__ == '__main__':
	le=len(sys.argv)
	if le==2:
		process(sys.argv[1])
	