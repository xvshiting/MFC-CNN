import numpy
from PIL import Image
import binascii
import os
import sys
import random 
import shutil  
image_path="C:\\Users\\intern_v\\Desktop\\image"



def crop_left_up(picture):
    im=picture.crop((0,0,32,32))
    print(im.size)
    return im
def crop_right_up(picture):
#     print(picture.size)
    im=picture.crop((picture.size[0]-32,picture.size[1]-32,picture.size[0],picture.size[1]))
    print(im.size)
    return im
def crop_left_bottom(picture):
    im=picture.crop((0,picture.size[1]-32,32,picture.size[1]))
    print(im.size)
    return im
def crop_right_bottom(picture):
    im=picture.crop((picture.size[0]-32,picture.size[1]-32,picture.size[0],picture.size[1]))
    print(im.size)
    return im
def crop_middle(picture):
    im=picture.crop(((picture.size[0]-32)/2,(picture.size[1]-32)/2,(picture.size[0]-32)/2+32,(picture.size[1]-32)/2+32))
    print(im.size)
    return im


def getMatrix_From_Bin(filename,width):
    with open(filename,'rb')as f:
        content=f.read()
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

def getfile(dir_path):
    try:
        pathDir =  os.listdir(dir_path)
        child=[]
        for allDir in pathDir:
            child.append(os.path.join('%s%s%s' % (dir_path,os.sep, allDir)))
        return child
    except:
        print("skip :"+str(dir_path))
def get_file_path(dir_path,image_path=image_path):
    class_name=str.split(dir_path,os.sep)
    file_dir=os.path.join(image_path,class_name[-1])
    try:
        os.listdir(image_path)
    except:
        os.mkdir(image_path)
        print("create new folder "+image_path)
    return file_dir

def save_file(image,file_dir,image_name):
    try:
        os.listdir(file_dir)
    except:
        info=sys.exc_info() 
        os.mkdir(file_dir)
        print("create new folder "+file_dir)
    image_path=os.path.join(file_dir,image_name)
    image.save(image_path+".png")
    print("save image:"+image_path+".png")
    return 0

def transfer(file_path,image_path,big_size=32,aug_data=False):
    raw_files=getfile(file_path)
    if raw_files==None: return 0
    print(len(raw_files))
    save_path=get_file_path(file_path,image_path)
    i=0
    for file in raw_files:
        # size=os.path.getsize(file)/1024
        # index=lambda x:1 if x<=10 else 2 if x<=30 else 3 if x<=60 else 4 if x<=100 else 5 if x<=200 else 6 if x<=500 else 7 if x<=1000 else 8
        # dic={1:32,2:64,3:128,4:256,5:284,6:384,7:512,8:768,9:1024}
        # width=dic[index(size)]
        images=[]
        im=Image.fromarray(getMatrix_From_Bin(file,1024))
        if aug_data==True and big_size!=32:
            im=im.resize((big_size,big_size),Image.ANTIALIAS)
            images.append(crop_left_up(im))
            images.append(crop_right_up(im))
            images.append(crop_right_bottom(im))
            images.append(crop_left_bottom(im))
            images.append(crop_middle(im))
        else:
            im=im.resize((32,32),Image.ANTIALIAS)
            images.append(im)
        for image in images:
            save_file(image,save_path,str.split(file,os.sep)[-1]+"_"+str(big_size)+"_"+str(int(random.random()*10000//1))+str(big_size))
        i=i+1
        if i%10==0:
            print(str(i))
            print(str.split(file,os.sep)[-1])
def get_image_path(file_path):
    print('......')
    return os.sep.join(str.split(file_path,os.sep)[:-1])+os.sep+"image"
def  create_image(big_size=32,aug_data=False):
    file_path="./sample"
    image_path=get_image_path(file_path)
    cu_image_dir=os.listdir(image_path)
    for image_dir in  cu_image_dir:
        try:
            shutil.rmtree(image_path+"/"+image_dir)
        except:
            os.remove(image_path+"/"+image_dir)
    for file in os.listdir(file_path):
                transfer(os.path.join(file_path,file),image_path,big_size,aug_data)



if __name__ == '__main__':
    file_path=sys.argv[1]
    aug=0
    aug_bool=False
    le=len(sys.argv)
    image_path=get_image_path(file_path)
    if le==3:
        aug=int(sys.argv[2])
# print(" Parameter must be one,check your file path!!!")
    else:
        print('................')
    if aug==1: 
        aug_bool=True
        print(image_path)
        sizes=range(33,45)
        for size in sizes:
            for file in os.listdir(file_path):
                transfer(os.path.join(file_path,file),image_path,size,aug_bool)
    else:
        for file in os.listdir(file_path):
                transfer(os.path.join(file_path,file),image_path,32,aug_bool)

# image_path="C:\\Users\\intern_v\\Desktop\\image"
# dir_path="C:\\Users\\intern_v\\Desktop\\sample"
# for file in os.listdir(dir_path):
#     transfer(os.path.join(dir_path,file),image_path)
#file=os.listdir(dir_path)[-1]
#transfer(os.path.join(dir_path,file))