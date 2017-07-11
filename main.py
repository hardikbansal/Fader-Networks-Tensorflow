import tensorflow as tf
import numpy as np
import optparse
import os
import shutil
import time
import random
import sys
import pickle
import glob

from layers import *

from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from scipy.misc import imresize
from PIL import Image
from tqdm import tqdm


class Fader():

	def run_parser(self):

		self.parser = optparse.OptionParser()

		self.parser.add_option('--num_iter', type='int', default=1000, dest='num_iter')
		self.parser.add_option('--batch_size', type='int', default=32, dest='batch_size')
		self.parser.add_option('--img_width', type='int', default=256, dest='img_width')
		self.parser.add_option('--img_height', type='int', default=256, dest='img_height')
		self.parser.add_option('--img_depth', type='int', default=3, dest='img_depth')
		self.parser.add_option('--num_attr', type='int', default=40, dest='num_attr')
		self.parser.add_option('--max_epoch', type='int', default=20, dest='max_epoch')
		self.parser.add_option('--num_train_images', type='int', default=500, dest='num_train_images')
		self.parser.add_option('--num_test_images', type='int', default=50, dest='num_test_images')
		self.parser.add_option('--test', action="store_true", default=False, dest="test")
		self.parser.add_option('--steps', type='int', default=10, dest='steps')
		self.parser.add_option('--enc_size', type='int', default=256, dest='enc_size')
		self.parser.add_option('--dec_size', type='int', default=256, dest='dec_size')
		self.parser.add_option('--model', type='string', default="draw_attn", dest='model_type')
		self.parser.add_option('--dataset', type='string', default="celebA", dest='dataset')
		self.parser.add_option('--dataset_dir', type='string', default="../datasets/img_align_celeba", dest='dataset_dir')
		self.parser.add_option('--test_dataset_dir', type='string', default="../datasets/img_align_celeba", dest='test_dataset_dir')


	def initialize(self):

		self.run_parser()

		opt = self.parser.parse_args()[0]

		self.max_epoch = opt.max_epoch
		self.batch_size = opt.batch_size
		self.dataset = opt.dataset

		self.img_width = opt.img_width
		self.img_height = opt.img_height
		self.img_depth = opt.img_depth

		self.img_size = self.img_width*self.img_height*self.img_depth
		self.num_attr = opt.num_attr
		self.num_train_images = opt.num_train_images
		self.num_test_images = opt.num_test_images
		self.model = "Fader"
		self.to_test = opt.test
		self.load_checkpoint = False
		self.do_setup = True
		self.dataset_dir = opt.dataset_dir

		self.tensorboard_dir = "./output/" + self.model + "/" + self.dataset + "/tensorboard"
		self.check_dir = "./output/"+ self.model + "/" + self.dataset +"/checkpoints"
		self.images_dir = "./output/" + self.model + "/" + self.dataset + "/imgs"

	def normalize_input(self, imgs):
		return imgs/127.5-1.0


	def load_dataset(self, mode='train'):

		if(mode == "train"):

			self.train_attr = []

			imageFolderPath = self.dataset_dir
			self.imagePath = glob.glob(imageFolderPath+'/*.jpg')


			dictn = []

			count = 0
			with open(self.dataset_dir+"/train_attr.txt") as f:
				for lines in f:
					temp = lines
					temp = temp.split()
					dictn.append(temp[1:])

			for i in range(self.num_train_images):
				self.train_attr.append(np.array(dictn[i]))

		elif (mode == "test"):

			self.test_attr = []

			imageFolderPath = self.test_dataset_dir
			self.imagePath = glob.glob(imageFolderPath+'/*.jpg')

			dictn = []

			count = 0
			with open(self.dataset_dir+"/test_attr.txt") as f:
				for lines in f:
					temp = lines
					temp = temp.split()
					dictn.append(temp[1:])

			for i in range(self.num_test_images):
				self.test_attr.append(np.array(dictn[i]))

	
	def load_batch(self, batch_num, batch_sz, mode="train"):

		if(mode == "train"):
			temp = []
			for i in range(batch_sz):
				temp.append(self.normalize_input(imresize(np.array(Image.open(self.imagePath[i + batch_sz*(batch_num)]),'f')[:,39:216,:], size=[256,256,3], interp="bilinear")))
			return temp

		elif (mode == "test"):
			temp = []
			for i in range(batch_sz):
				temp.append(self.normalize_input(imresize(np.array(Image.open(self.imagePath[i + batch_sz*(batch_num)]),'f')[:,39:216,:], size=[256,256,3], interp="bilinear")))
			return temp

	def generation_loss(self, input_img, output_img, loss_type='mse'):

		if (loss_type == 'mse'):
			return tf.reduce_sum(tf.squared_difference(input_img, output_img), [1, 2, 3])
		elif (loss_type == 'log_diff'):
			epsilon = 1e-8
			return -tf.reduce_sum(input_img*tf.log(output_img+epsilon) + (1 - input_img)*tf.log(epsilon + 1 - output_img),[1, 2, 3])


	def discriminator_loss(self, out_attr, inp_attr):

		return tf.reduce_sum(tf.abs(out_attr-inp_attr),1)
	
	def encoder(self, input_enc, name="Encoder"):

		with tf.variable_scope(name) as scope:

			o_c1 = general_conv2d(input_enc, 16, name="C16")
			o_c2 = general_conv2d(o_c1, 32, name="C32")
			o_c3 = general_conv2d(o_c2, 64, name="C64")
			o_c4 = general_conv2d(o_c3, 128, name="C128")
			o_c5 = general_conv2d(o_c4, 256, name="C256")
			o_c6 = general_conv2d(o_c5, 512, name="C512_1")
			o_c7 = general_conv2d(o_c6, 512, name="C512_2")

			return o_c7

	def decoder(self, input_dec, attr, name="Decoder"):

		with tf.variable_scope(name) as scope:

			attr_1 = tf.stack([attr]*4)
			o_d0 = tf.concat([input_dec, tf.reshape(tf.transpose(attr_1,[1, 0, 2]),[-1, 2, 2, 40])], 3)
			o_d1 = general_deconv2d(o_d0, 512, name="D512_2")
			
			attr_2 = tf.transpose(tf.stack([attr]*16),[1, 0, 2])
			o_d1 = tf.concat([o_d1, tf.reshape(tf.transpose(attr_2,[1, 0, 2]),[-1, 4, 4, 40])], 3)			
			o_d2 = general_deconv2d(o_d1, 256, name="D512_1")
			
			attr_3 = tf.transpose(tf.stack([attr]*64),[1, 0, 2])
			o_d2 = tf.concat([o_d2, tf.reshape(tf.transpose(attr_3,[1, 0, 2]),[-1, 8, 8, 40])], 3)
			o_d3 = general_deconv2d(o_d2, 128, name="D256")
			
			attr_4 = tf.transpose(tf.stack([attr]*256),[1, 0, 2])
			o_d3 = tf.concat([o_d3, tf.reshape(tf.transpose(attr_4,[1, 0, 2]),[-1, 16, 16, 40])], 3)
			o_d4 = general_deconv2d(o_d3, 64, name="D128")
			
			attr_5 = tf.transpose(tf.stack([attr]*1024),[1, 0, 2])
			o_d4 = tf.concat([o_d4, tf.reshape(tf.transpose(attr_5,[1, 0, 2]),[-1, 32, 32, 40])], 3)
			o_d5 = general_deconv2d(o_d4, 32, name="D64")
			
			attr_6 = tf.transpose(tf.stack([attr]*4096),[1, 0, 2])
			o_d5 = tf.concat([o_d5, tf.reshape(tf.transpose(attr_6,[1, 0, 2]),[-1, 64, 64, 40])], 3)
			o_d6 = general_deconv2d(o_d5, 16, name="D32")
			

			attr_7 = tf.transpose(tf.stack([attr]*16384),[1, 0, 2])
			o_d6 = tf.concat([o_d6, tf.reshape(tf.transpose(attr_7,[1, 0, 2]),[-1, 128, 128, 40])], 3)
			o_d7 = general_deconv2d(o_d6, 3, name="D16")

			return o_d7


	def discriminator(self, input_disc, name="Discriminator"):

		with tf.variable_scope(name) as scope:

			o_disc1 = general_conv2d(input_disc, 512, name="C512")
			size_disc = o_disc1.get_shape().as_list()
			o_flat = tf.reshape(o_disc1,[self.batch_size, 512])
			o_disc2 = linear1d(o_flat, 512, 512, name="fc1")
			o_disc3 = linear1d(o_disc2, 512, self.num_attr, name="fc2")

			return tf.nn.sigmoid(o_disc3)
			

	def celeb_model_setup(self):

		self.input_imgs = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth])
		self.input_attr = tf.placeholder(tf.float32, [self.batch_size, self.num_attr])
		self.lmda = tf.placeholder(tf.float32,[1])

		with tf.variable_scope("Model") as scope:

			self.o_enc = self.encoder(self.input_imgs)
			self.o_dec = self.decoder(self.o_enc, self.input_attr)
			self.o_disc = self.discriminator(self.o_enc)
			


	def model_setup(self):

		with tf.variable_scope("Model") as scope:
			self.celeb_model_setup()

		self.model_vars = tf.trainable_variables()
		for var in self.model_vars: print(var.name, var.get_shape())

		self.do_setup = False

	def loss_setup(self):

		self.img_loss = tf.reduce_mean(self.generation_loss(self.input_imgs, self.o_dec))
		self.enc_loss = tf.reduce_mean(self.discriminator_loss(self.o_disc, 1-self.input_attr))
		
		self.disc_loss = tf.reduce_mean(self.discriminator_loss(self.o_disc, self.input_attr))
		self.enc_dec_loss = self.img_loss + self.lmda*self.enc_loss

		optimizer = tf.train.AdamOptimizer(0.002, beta1=0.5)

		enc_dec_vars = [var for var in self.model_vars if 'coder' in var.name]
		disc_vars = [var for var in self.model_vars if 'Discriminator' in var.name]

		self.enc_dec_loss_optimizer = optimizer.minimize(self.enc_dec_loss, var_list=enc_dec_vars)
		self.disc_loss_optimizer = optimizer.minimize(self.disc_loss, var_list=disc_vars)

	def train(self):

		self.model_setup()
		self.loss_setup()

		self.load_dataset()

		# sys.exit()

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()

		if not os.path.exists(self.images_dir+"/train/"):
			os.makedirs(self.images_dir+"/train/")
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)


		with tf.Session() as sess:

			sess.run(init)
			writer = tf.summary.FileWriter(self.tensorboard_dir)
			writer.add_graph(sess.graph)

			if self.load_checkpoint:
				chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
				saver.restore(sess,chkpt_fname)

			for epoch in range(0, self.max_epoch):

				for itr in range(0, int(self.num_train_images/self.batch_size)):

					temp_lmd = 0.0001*(epoch*self.batch_size + itr)/(self.batch_size*self.max_epoch)

					imgs = self.load_batch(itr, self.batch_size)
					attrs = self.train_attr[itr*self.batch_size:(itr+1)*(self.batch_size)]

					_, temp_tot_loss, temp_img_loss, temp_enc_loss = sess.run(
						[self.enc_dec_loss_optimizer, self.enc_dec_loss, self.img_loss, self.enc_loss],
						feed_dict={self.input_imgs:imgs, self.input_attr:attrs, self.lmda:temp_lmd})


					_, temp_disc_loss = sess.run(
						[self.disc_loss_optimizer, self.disc_loss],
						feed_dict={self.input_imgs:imgs, self.input_attr:attrs, self.lmda:temp_lmd})

					print("We are in epoch "+ str(epoch) + " with a total_loss of " + str(temp_tot_loss) +
					 " image_loss of " + str(temp_img_loss) + " and discriminator_loss of " + str(temp_disc_loss))

				saver.save(sess,os.path.join(check_dir,"Fader"),global_step=epoch)

	def test(self):

		self.model_setup()
		self.load_dataset(mode="test")

		if not os.path.exists(self.images_dir+"/test/"):
			os.makedirs(self.images_dir+"/test/")
		if not os.path.exists(self.check_dir):
			os.makedirs(self.check_dir)



		with tf.Session() as sess:

			chkpt_fname = tf.train.latest_checkpoint(self.check_dir)
			saver.restore(sess, chkpt_fname)

			for itr in range(0, int(self.num_test_images/self.batch_size)):

				imgs = self.load_batch(itr, self.batch_size, mode="test")
				attrs = self.test_attr[itr*self.batch_size:(itr+1)*(self.batch_size)]

				temp_output = sess.run([self.o_dec], feed_dict={self.input_imgs:imgs, self.input_attr:attrs})




def main():

	model = Fader()
	model.initialize()

	if(model.to_test):
		model.test()
	else:
		model.train()


if __name__ == "__main__":
	main()