#!/usr/bin/python

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
	output_dir = '/home/hack17/joe/raw/'

print('Convert {}*.jpg files to raw and place it in {}'.format(input_dir, output_dir))

addrs=[]
tmp_output_dir = output_dir
for root, subFolders, files in os.walk(input_dir):
 	filename=root+'/*.jpg'
	newpath=root.split(input_dir)[1]
	if not os.path.exists(output_dir+'/'+newpath+'/'):
	    print("creating DIR " + newpath)
	    os.makedirs(output_dir+'/'+newpath)
	cat_dog_train_path = filename
 	print(filename)
	addrs = addrs + glob.glob(cat_dog_train_path)

labels = [0 if 'cat' in addr else 1 for addr in addrs]

shuffle_data = True
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

train_addrs = addrs[0:int(0.001*len(addrs))]
train_labels = labels[0:int(0.001*len(labels))]

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

for i in range(len(train_addrs)):
	newpath=train_addrs[i].split(input_dir)[1]
	fileName = train_addrs[i].rsplit('/',1)[1].rsplit('.',1)[0]
	subpath = newpath.split(fileName)[0] + '/'
 	outFile = output_dir + subpath + fileName + '.raw'
	print(outFile)
	with open(outFile, "wb") as output_file:
	    # print how many images are saved every 1000 images
	    if not i % 1000:
		print 'Convert data: {}/{}'.format(i, len(train_addrs))
		sys.stdout.flush()
	    # Load the image
	    img = load_image(train_addrs[i])
            cPickle.dump(img, output_file)

print('Conversion Done!!')

