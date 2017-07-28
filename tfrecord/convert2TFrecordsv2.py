#!/usr/bin/python

import tensorflow as tf
import glob
import numpy as np
import cv2
import sys
import cPickle
from random import shuffle
import sys
import os

if len(sys.argv) > 2:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
else:
        input_dir = '/home/hack17/cd/train/'
        output_dir = '/home/hack17/joe/tfrecord/'

print('Convert {}*.jpg files to TFRecord and place it in {}'.format(input_dir, output_dir))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

addrs=[]
tmp_output_dir = output_dir
for root, subFolders, files in os.walk(input_dir):
        filename=root+'/*.jpg'
        cat_dog_train_path = filename
        addrs = addrs + glob.glob(cat_dog_train_path)

# Read labels
labelNames = []
with open(input_dir + '/labels.txt') as f:
	labelNames =  f.readlines()

labelNames =  [x.strip() for x in labelNames]

print(labelNames)

# labels = [0 if 'cat' in addr else 1 for addr in addrs]
labels = [ labelNames.index(lab) for lab in labelNames  for addr in addrs if lab in addr ]

shuffle_data = True

# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.01*len(addrs))]
train_labels = labels[0:int(0.01*len(labels))]



def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_filename = output_dir+ '/' + 'train-00000-of-00001' 
writer = tf.python_io.TFRecordWriter(train_filename)


for i in range(len(train_addrs)):
    # print how many images are saved every 1000 images
    if not i % 1000:
	print 'Convert data: {}/{}'.format(i, len(train_addrs))
	sys.stdout.flush()
    # Load the image
    # img = load_image(train_addrs[i])

    label = train_labels[i]
    with tf.gfile.FastGFile(train_addrs[i], 'rb') as f:
    	image_data = f.read()
    with tf.Session() as sess:
	_decode_jpeg_data = tf.placeholder(dtype=tf.string)
	_decode_jpeg = tf.image.decode_jpeg(_decode_jpeg_data, channels=3)
	image_buffer = sess.run(_decode_jpeg,
                           feed_dict={_decode_jpeg_data: image_data})


    colorspace = 'RGB'
    channels = 3
    image_format = 'JPEG'
    height = image_buffer.shape[0]
    width = image_buffer.shape[1]
    text = "dog" if label == 1 else "cat"

    # Create a feature
    feature={ 'image/height': _int64_feature(height),
		'image/width': _int64_feature(width),
		'image/colorspace': _bytes_feature(tf.compat.as_bytes(colorspace)),
		'image/channels': _int64_feature(channels),
		'image/class/label': _int64_feature(label),
		'image/class/text': _bytes_feature(tf.compat.as_bytes(text)),
		'image/format': _bytes_feature(tf.compat.as_bytes(image_format)),
		'image/filename': _bytes_feature(tf.compat.as_bytes(train_addrs[i])),
		'image/encoded': _bytes_feature(tf.compat.as_bytes(image_data))}

    # Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()
sys.stdout.flush()


print("Dump the TFRecords into a file")


