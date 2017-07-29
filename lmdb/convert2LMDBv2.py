#!/usr/bin/python

import numpy as np
import cv2
import sys
import os
import glob
from subprocess import call
from random import shuffle
import sys
import os
import re
from commonutils import cmd_exists

# data for train & validate
FRACTION=0.5

if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
	prefix = sys.argv[3]
else:
        input_dir = '/home/hack17/cd/train/'
        output_dir = '/home/hack17/joe/lmdb/'
	prefix="dogcat"

if (not cmd_exists("convert_imageset") or not cmd_exists("compute_image_mean")):
	print("Make sure caffe tools are in PATH")
	sys.exit()

DATA_DIR=input_dir
TXT_DIR=output_dir

print('Convert {}*.jpg files to LMDB and place it in {}'.format(input_dir, output_dir))

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read labels
labelNames = []
with open(input_dir + '/labels.txt') as f:
        labelNames =  f.readlines()

labelNames =  [x.strip() for x in labelNames]

print(labelNames)

addrs=[]
tmp_output_dir = output_dir
for root, subFolders, files in os.walk(input_dir):
        filename=root+'/*.jpg'
        cat_dog_train_path = filename
        addrs = addrs + glob.glob(cat_dog_train_path)

# labels = [0 if 'cat' in addr else 1 for addr in addrs]
labels = [ labelNames.index(lab) for lab in labelNames  for addr in addrs if lab in addr ]

shuffle_data = True
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)


train_addrs = addrs[:int(FRACTION*len(addrs))]
train_labels = labels[:int(FRACTION*len(labels))]
val_addrs = addrs[int((1-FRACTION)*len(addrs)):]
val_labels = labels[int((1-FRACTION)*len(labels)):]

train_addrs_labels = zip(train_addrs, val_labels)
val_addrs_labels = zip(val_addrs, val_labels)

with open('{}/train.txt'.format(TXT_DIR), 'w') as f:
    for image,label in train_addrs_labels:
        f.write('{} {}\n'.format(image.split(input_dir)[1], label))
    f.close()
 
with open('{}/test.txt'.format(TXT_DIR), 'w') as f:
    for image,label in val_addrs_labels:
        f.write('{} {}\n'.format(image.split(input_dir)[1], label))
    f.close()


inputdir=TXT_DIR
outpdir=TXT_DIR
traindata=DATA_DIR
valdata=DATA_DIR

call(['bash', 'create_imagenet.sh', prefix, inputdir, outpdir, traindata, valdata])

print("Conversion complete. LMDB files can be found under {} !!".format(outpdir))

call(['bash', 'make_imagenet_mean.sh', prefix, inputdir, outpdir])

print("Mean image file created under {} !!".format(outpdir))


dict = {
    "OUTPUT_DIR" : outpdir,
    "NUM_LABELS" : str(len(labelNames)),
    "PREFIX"     : prefix
} 

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

path = os.path.dirname(os.path.realpath(__file__))

with open(path+"/train_val.prototxt.template") as text:
    new_text = multiple_replace(dict, text.read())
with open(outpdir+"/train_val.prototxt", "w") as result:
    result.write(new_text)

with open(path+"/solver.prototxt.template") as text:
    new_text = multiple_replace(dict, text.read())
with open(outpdir+"/solver.prototxt", "w") as result:
    result.write(new_text)

print("train_val.prototxt & solver.prototxt files are created under {} !!".format(outpdir))



