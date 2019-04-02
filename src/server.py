import tornado.ioloop
import tornado.web
import predict
import training
import load_data
class re_training(tornado.web.RequestHandler):
 	def get(self):
 		# self.write("Start traning.......")
 		# self.write("Traning a whole new model , without new data")
 		training.retrain()
 		self.write("End traning")
class training_all_new(tornado.web.RequestHandler):
 	def get(self):
 		self.write("Start traning.......")
 		self.write("Traning a whole new model , Creat new data")
 		training.train()
 		self.write("End traning")
class training_all_new_with_aug(tornado.web.RequestHandler):
 	def get(self):
 		self.write("Start traning.......")
 		self.write("Traning a whole new model  , Creat new data with data augmentation!")
 		training.train_With_Aug()
 		self.write("End traning")
class predic(tornado.web.RequestHandler):
 	def get(self):
 		self.write("hahah")
 	def post(self):
 		file=self.request.body
 		# print(len(file))
 		# file=self.get_argument("sample")
 		self.write(predict.predict(file))
class getlist(tornado.web.RequestHandler):
	def get(self):
		self.write(load_data.get_list())
def make_app():
 	return tornado.web.Application([
 		(r"/predict",predic),
 		(r"/re_training",re_training),
 		(r"/training",training_all_new),
 		(r"/get_list",getlist),
 		(r"/trainingwithaug", training_all_new_with_aug),
 		])

if __name__ == '__main__':
	# parse_options()
	# http_server=tornado.httpserver.HTTPServer(Application(),xheaders=True)
 	app=make_app()
 	app.listen(8888,address="0.0.0.0")
 	tornado.ioloop.IOLoop.current().start()