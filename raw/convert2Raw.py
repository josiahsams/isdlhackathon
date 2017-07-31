#!/usr/bin/python

import glob
import numpy as np
import cv2
import sys
import cPickle
from random import shuffle
import sys
import os

# data split for train & validate
SPLIT=0.5

# data fraction for experiment purpose
FRACTION=0.01

if len(sys.argv) > 2:
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
else:
	input_dir = '/home/hack17/cd/train/'
	output_dir = '/home/hack17/joe/raw/'

if len(sys.argv) > 3:
        SPLIT=float(sys.argv[3])

if len(sys.argv) > 4:
        FRACTION=float(sys.argv[4])


if SPLIT > 1.0:
        print("SPLIT can not be greater than 1.0")
        sys.exit()

if FRACTION > 1.0:
        print("FRACTION can not be greater than 1.0")
        sys.exit()

print('Convert {}*.jpg files to raw and place it in {}'.format(input_dir, output_dir))

addrs=[]
tmp_output_dir = output_dir
for root, subFolders, files in os.walk(input_dir):
 	filename=root+'/*.jpg'
	newpath=root.split(input_dir)[1]
	if not os.path.exists(output_dir+'/'+newpath+'/'):
	    os.makedirs(output_dir+'/'+newpath)
	cat_dog_train_path = filename
	addrs = addrs + glob.glob(cat_dog_train_path)

# Read labels
labelNames = []
with open(input_dir + '/labels.txt') as f:
        labelNames =  f.readlines()

labelNames =  [x.strip() for x in labelNames]

print(labelNames)

labels = [ labelNames.index(lab) for lab in labelNames  for addr in addrs if lab in addr ]

shuffle_data = True
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

print("No. of samples selected for Train is {} out of {}".format(int(SPLIT*FRACTION*len(addrs)), len(addrs)))
print("No. of samples selected for Validation is {} out of {}".format(len(addrs)-int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)), len(addrs)))

train_addrs = addrs[0:int(SPLIT*FRACTION*len(addrs))]
train_labels = labels[0:int(SPLIT*FRACTION*len(labels))]

val_addrs = addrs[int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)):]
val_labels = labels[int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)):]

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img

with open(output_dir+"/train.txt", "w") as trainfile:
    for i in range(len(train_addrs)):
	newpath=train_addrs[i].split(input_dir)[1]
	fileName = train_addrs[i].rsplit('/',1)[1].rsplit('.',1)[0]
	subpath = newpath.split(fileName)[0] + '/'
 	outFile = output_dir + subpath + fileName + '.raw'
	with open(outFile, "wb") as output_file:
	    # print how many images are saved every 1000 images
	    if not i % 1000:
		print 'Convert data: {}/{}'.format(i, len(train_addrs))
		sys.stdout.flush()
            trainfile.write('{} {}\n'.format(subpath+ fileName + '.raw', train_labels[i]))
	    # Load the image
	    img = load_image(train_addrs[i])
            cPickle.dump(img, output_file)


with open(output_dir+"/validate.txt", "w") as valfile:
    for i in range(len(val_addrs)):
        newpath=val_addrs[i].split(input_dir)[1]
        fileName = val_addrs[i].rsplit('/',1)[1].rsplit('.',1)[0]
        subpath = newpath.split(fileName)[0] + '/'
        outFile = output_dir + subpath + fileName + '.raw'
        valfile.write('{} {}\n'.format(subpath+ fileName + '.raw', val_labels[i]))
        with open(outFile, "wb") as output_file:
            # print how many images are saved every 1000 images
            if not i % 1000:
                print 'Convert data: {}/{}'.format(i, len(val_addrs))
                sys.stdout.flush()
            # Load the image
            img = load_image(val_addrs[i])
            cPickle.dump(img, output_file)

print('Conversion Done!!')

