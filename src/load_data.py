import numpy as np
import pickle as pk
import os
class DataSet(object):
    def __init__(self,images,label):
        images = images.astype(np.float32)
        images = np.multiply(images, 1.0 / 255.0)
        self._images=images
        self._labels=label
        self._epochs_completed=0
        self._index_in_epoch=0
        self._num_examples = images.shape[0]
    @property
    def images(self):
        return self._images
    @property
    def labels(self):
        return self._labels
    @property
    def num_example(self):
        return self._num_examples
    @property
    def epochs_completed(self):
        return self._epochs_completed
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        # print(self._num_examples)
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            start = 0
            self._index_in_epoch = batch_size
            # print(batch_size,self._num_examples)
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
def get_list():
    input_data=open("data"+os.sep+"data.pkl","rb")
    cl_data=pk.load(input_data)
    cl_map=cl_data[0]
    stt="Virus family  number  :    "+str(cl_data[1])+"\n"
    i=1
    for key,value in cl_map.items():
        stt=stt+key+"      "+str(value)
        # if  1%2==0:
        stt=stt+"\n"
    return stt
def read_data_set():
    input_image=open("data"+os.sep+"image_data.pkl","rb")
    input_label=open("data"+os.sep+"image_label_data.pkl","rb")
    input_data=open("data"+os.sep+"data.pkl","rb")
    image_data=np.array(pk.load(input_image))
    input_image.close()
    lable_data=np.array(pk.load(input_label))
    input_label.close()
    cl_data=pk.load(input_data)
    input_data.close()
    # print(cl_data)
    cl_num=cl_data[1]
    cl_map=cl_data[0]
    class DataSets(object):
        def __init__(self):
            self.maps=cl_map
            self.nums=cl_num
    data_sets = DataSets()
    #分割数据
    sample_num=image_data.shape[0]
    train_start=0
    train_end=int(sample_num*3/6)
    test_start=train_end+1
    test_end=int(sample_num*5/6)
    validation_start=test_end+1
    validation_end=sample_num-1
    print(train_start,train_end,test_start,test_end,validation_start,validation_end)
    data_sets.train=DataSet(images=image_data[train_start:sample_num-1],label=lable_data[train_start:sample_num-1])
    data_sets.test=DataSet(images=image_data[test_start:test_end],label=lable_data[test_start:test_end])
    data_sets.validation=DataSet(images=image_data[validation_start:validation_end],label=lable_data[validation_start:validation_end])
    return data_sets