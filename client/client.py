import urllib
from urllib import request,parse
import sys
import zlib
my_url="http://localhost:8888/"
def  con_sever(re_url,data=None):
	if data:
		req=request.Request(url=re_url,data=data)
	else :
		req=request.Request(url=re_url)
	proxy_support=request.ProxyHandler({})
	opener=request.build_opener(proxy_support)
	# print(opener.open(req).read().decode("utf-8"))
	print(opener.open(req).read().decode("utf-8"))

def  predict(file_path):
	content=""
	re_url=my_url+"predict"
	with open(file_path,'rb')as f:
		content=f.read()
	con_sever(re_url=re_url,data=content)
def  traning_without_new_data():
	re_url=my_url+"re_training"
	con_sever(re_url=re_url)
	# "/re_training",training),
 # 		(r"/training"
def  traning_with_new_data():
	re_url=my_url+"training"
	con_sever(re_url=re_url)
def  get_list():
	re_url=my_url+"get_list"
	con_sever(re_url=re_url)
def training_all_new_with_aug():
	re_url=my_url+"trainingwithaug"
	con_sever(my_url)
if __name__ == '__main__':
	chioice=sys.argv[1]
	if  chioice=="-rt":
		print("Start traning.......")
		print("Traning a whole new model , without new data")
		traning_without_new_data() 
	if  chioice=="-p":
		file=sys.argv[2]
		predict(file)
	if chioice=="-t":
		print("Start traning.......")
		print("Traning a whole new model , with new data")
		traning_with_new_data()
	if chioice=="-l":
		get_list()
	if chioice=="-ta":
		training_all_new_with_aug()
	if chioice=="--help":
		print("-l   :    get the name of virus family we can classify")
		print("-p  :    prediction !  need  a Filepath  ")
		print("-t    :    Train with New data!")
		print("-rt  :    Train without New data!")
