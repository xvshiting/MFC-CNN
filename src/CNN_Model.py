import sys
import tensorflow as tf
import numpy as np
import os
sys.path.append("./src")
import load_data
import pickle as pk
import shutil
input_data=open("data"+os.sep+"data.pkl","rb")
cl_data=pk.load(input_data)
cl_num=cl_data[1]
cl_map=cl_data[0]
cl_map_1=dict()
for key,value in cl_map.items():
	cl_map_1[value]=key
def  weight_variable(shape,name,wl=None,stddev=0.1):
    var =tf.Variable(tf.truncated_normal(shape,stddev=0.1),name=name,dtype=tf.float32)
    if wl is not None:
        weight_loss=tf.multiply(tf.nn.l2_loss(var),wl,name="weight_loss")
        tf.add_to_collection('losses',weight_loss)
    return var
def bias_variable(shape,name):
    initial=tf.Variable(tf.constant(0.1,shape=shape),name=name,dtype=tf.float32)
    return initial

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def train(cls_num=cl_num):
	tf.reset_default_graph()
	virus=load_data.read_data_set()
	learning_rate=tf.Variable(0.00001,name="learning_rate")
	reduce_rate=tf.constant(-0.0000000001,name="reduce_rate")
	new_rate=tf.add(learning_rate,reduce_rate)
	update_l_rate=tf.assign(learning_rate,new_rate)
	W_conv1=weight_variable([3,3,1,32],name="W_conv_1")
	b_conv1=bias_variable([32],name="b_conv1")
	W_conv1_1=weight_variable([3,3,32,32],name="W_conv1_1")
	b_conv1_1=bias_variable([32],name="b_conv1_1")
	x=tf.placeholder("float",[None,32*32])
	y_=tf.placeholder("float",[None,11])
	x_image=tf.reshape(x,[-1,32,32,1])
	h_cov1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
	h_cov1_1=tf.nn.relu(conv2d(h_cov1,W_conv1_1)+b_conv1_1)
	h_pool1=max_pool_2x2(h_cov1_1)
	W_conv2=weight_variable([3,3,32,64],name="W_conv2")
	b_conv2=bias_variable([64],name="b_conv2")
	W_conv2_1=weight_variable([3,3,64,64],name="W_conv2_1")
	b_conv2_1=bias_variable([64],name="b_conv2_1")
	h_cov2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_cov2_1=tf.nn.relu(conv2d(h_cov2,W_conv2_1)+b_conv2_1)
	h_pool2=max_pool_2x2(h_cov2_1)
	W_conv3=weight_variable([3,3,64,128],name="W_conv3")
	b_conv3=bias_variable([128],name="b_conv3")
	h_cov3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
	h_pool3=max_pool_2x2(h_cov3)
	w_fc1=weight_variable([4*4*128,1024],name="w_fc1")
	b_fc1=bias_variable([1024],name="b_fc1")
	h_pool2_flat=tf.reshape(h_pool3,[-1,4*4*128])
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
	keep_prob=tf.placeholder("float")
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
	w_fc2=weight_variable([1024,11],name="w_fc2")
	b_fc2=bias_variable([cls_num],name="b_fc2")
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
	cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
	# tf.add_to_collection('losses',cross_entropy)
	# loss=tf.add_n(tf.get_collection('losses'),name='total_loss')
	train_step=tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
	correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
	init=tf.global_variables_initializer()
	saver=tf.train.Saver()
	pre_accuracy=0  #early stopping
	with tf.Session() as sess:
		with tf.device("/cpu:0"):
			sess.run(init)
#     writer.add_graph(sess.graph)
			for i in range(50000):
				batch=virus.train.next_batch(500)
				if (i+1)%100==0:
					train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
					print("step %d,training accuracy %g" %(i,train_accuracy))
#         print(i)
#             s=sess.run(merger_summary,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
#             writer.add_summary(s,i)
					train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:1})

					if  pre_accuracy-train_accuracy>0.4:
						break
					pre_accuracy=train_accuracy
					save_path=saver.save(sess,"./training_model/CNN_model.ckpt")

					print ("model saved in file :", save_path)

					print("step %d,training accuracy %g" %(i,train_accuracy))
				sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:1})
				sess.run(update_l_rate)
	#mv  ./training_model/*.ckpt  -----> ./trained_model/*.ckpt
			files=os.listdir("./training_model")
			old_files=os.listdir("./trained_model")
			for file in old_files:
				os.remove("./trained_model/"+file)
			for file in files:
				old_path="./training_model/"+file
				new_path="./trained_model"
				shutil.move(old_path,new_path)  
			
#         if (i+1)%500==0:
#             print ("test accuracy %g" %sess.run(accuracy,feed_dict={x:virus.test.images,y_:virus.test.labels,keep_prob:0.8}))
#             print ("training accuracy %g" %sess.run(accuracy,feed_dict={x:virus.train.images,y_:virus.train.labels,keep_prob:0.8}))
#             print ("validation accuracy %g" %sess.run(accuracy,feed_dict={x:virus.validation.images,y_:virus.validation.labels,keep_prob:0.8}))
#     print ("test accuracy %g" %sess.run(accuracy,feed_dict={x:virus.test.images,y_:virus.test.labels,keep_prob:0.8}))
#     save_parame(b_conv1=b_conv1,b_conv2=b_conv2,w_fc1=w_fc1,w_fc2=w_fc2,W_conv1=W_conv1,W_conv2=W_conv2,b_fc1=b_fc1,b_fc2=b_fc2)
#     batch=load_parame()
#     for i in batch:
#         print(i.shape)
#     print ("training accuracy %g" %sess.run(accuracy,feed_dict={x:virus.train.images,y_:virus.train.labels,keep_prob:0.8}))
#     print ("validation accuracy %g" %sess.run(accuracy,feed_dict={x:virus.validation.images,y_:virus.validation.labels,keep_prob:0.8}))
def  predict(cls_num=cl_num,data=None):
	tf.reset_default_graph()
	learning_rate=tf.Variable(0.00001,name="learning_rate")
	reduce_rate=tf.constant(-0.0000000003,name="reduce_rate")
	new_rate=tf.add(learning_rate,reduce_rate)
	update_l_rate=tf.assign(learning_rate,new_rate)
	W_conv1=weight_variable([3,3,1,32],name="W_conv_1")
	b_conv1=bias_variable([32],name="b_conv1")
	W_conv1_1=weight_variable([3,3,32,32],name="W_conv1_1")
	b_conv1_1=bias_variable([32],name="b_conv1_1")
	x=tf.placeholder("float",[None,32*32])
	y_=tf.placeholder("float",[None,11])
	x_image=tf.reshape(x,[-1,32,32,1])
	h_cov1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
	h_cov1_1=tf.nn.relu(conv2d(h_cov1,W_conv1_1)+b_conv1_1)
	h_pool1=max_pool_2x2(h_cov1_1)
	W_conv2=weight_variable([3,3,32,64],name="W_conv2")
	b_conv2=bias_variable([64],name="b_conv2")
	W_conv2_1=weight_variable([3,3,64,64],name="W_conv2_1")
	b_conv2_1=bias_variable([64],name="b_conv2_1")
	h_cov2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_cov2_1=tf.nn.relu(conv2d(h_cov2,W_conv2_1)+b_conv2_1)
	h_pool2=max_pool_2x2(h_cov2_1)
	W_conv3=weight_variable([3,3,64,128],name="W_conv3")
	b_conv3=bias_variable([128],name="b_conv3")
	h_cov3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)
	h_pool3=max_pool_2x2(h_cov3)
	w_fc1=weight_variable([4*4*128,1024],name="w_fc1")
	b_fc1=bias_variable([1024],name="b_fc1")
	h_pool2_flat=tf.reshape(h_pool3,[-1,4*4*128])
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
	keep_prob=tf.placeholder("float")
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
	w_fc2=weight_variable([1024,11],name="w_fc2")
	b_fc2=bias_variable([cls_num],name="b_fc2")
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)
	saver=tf.train.Saver()
	pre_accuracy=0  #early stoppin
	sess=tf.Session()
	saver.restore(sess,"./trained_model/CNN_model.ckpt")
	s=sess.run(y_conv,feed_dict={x:data,keep_prob:1})
	c=[]
	p=[]
	print(s)
	for i in range(0,3):
		c.append(np.argmax(s))
		p.append(np.max(s))
		s[0][np.argmax(s)]=0
	strs=""
	for i,j in zip(c,p):
		strs=strs+"Class: "+cl_map_1[i+1]+"                   P" +":  "+str(round(j*1000/10))+"%"+"\n"
	return strs
# 	with tf.Session() as sess:
# 		saver.restore(sess,"./trained_model/CNN_model.ckpt")
# 		for i in range(1000):
# 			batch=virus.train.next_batch(100)
# #         print(i)
# 			if (i+1)%100==0:
# #             s=sess.run(merger_summary,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
# #             writer.add_summary(s,i)
# 				train_accuracy=sess.run(accuracy,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
# 				if  pre_accuracy-train_accuracy>0.4:
# 					break

# 				print("step %d,training accuracy %g" %(i,train_accuracy))
# 			sess.run(train_step,feed_dict={x:batch[0],y_:batch[1],keep_prob:0.8})
# 			sess.run(update_l_rate)

		
if __name__ == '__main__':
	# train()
	sess=predict()
	print(sess)

