import random

import cv2
import time

import utils
import os
from argparse import ArgumentParser
from os.path import exists

import numpy as np
import tensorflow as tf

import transform


cap = cv2.VideoCapture(-1)
count = 0

ckpoints =[ '/home/iamukasa/PythonProjects/Fastdemo/checkpoint/starry',
            "/home/iamukasa/PythonProjects/Fastdemo/checkpoint/princess",
           "/home/iamukasa/PythonProjects/Fastdemo/checkpoint/dorra"]



def ffwd(content, network_path,cin):
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        img_placeholder = tf.placeholder(tf.float32, shape=content.shape,
                                         name='img_placeholder')
        network = transform.net(img_placeholder)
        saver=tf.train.Saver()

        saver.restore(sess,tf.train.latest_checkpoint(network_path))


        prediction = sess.run(network, feed_dict={img_placeholder: content})


        sess.close()
        return prediction[0]


def getstyled(data_in, cin):
    #str(time.time())
    paths_out = "styled/test" + str(time.time())+ str(cin) + ".jpg"
    # content_image = utils.load_image(data_in)
    content_image=data_in
    reshaped_content_height = (content_image.shape[0] - content_image.shape[0] % 4)
    reshaped_content_width = (content_image.shape[1] - content_image.shape[1] % 4)
    reshaped_content_image = content_image[:reshaped_content_height, :reshaped_content_width, :]
    reshaped_content_image = np.ndarray.reshape(reshaped_content_image, (1,) + reshaped_content_image.shape)

    try:
        prediction = ffwd(reshaped_content_image, random.choice(ckpoints), cin)
        utils.save_image(prediction, paths_out)
        print(prediction)
        return prediction,paths_out
    except:
        prediction = ffwd(reshaped_content_image, random.choice(ckpoints), cin)
        utils.save_image(prediction, paths_out)
        print(prediction,)
        return prediction,paths_out

if cap.isOpened():
    rval,frame=cap.read()
else :
    rval=False

while True:
    #cv2.imshow("Inception Classifier", frame)
    # true or false for ret if the capture is there or not
    ret, frame = cap.read()  # read frame from the webcasm

    print(frame)
    #
    # name_of_file="frames/frame%d.jpg" % count
    # cv2.imwrite(name_of_file,frame)
    #if count ==1:
    #   time.sleep(3000000)
    newframe,path=getstyled(frame,count)


    utils.load_image(path)
    #cv2.imshow("Style Transfer",newframe)


    count +=1


    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyWindow("Inception classifier")