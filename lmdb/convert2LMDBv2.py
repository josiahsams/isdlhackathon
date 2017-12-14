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

# image size after resize ;
# resized images are used for training & testing
img_height = 227
img_width = 227

# data split for train & validate
SPLIT=0.5

# data fraction for experiment purpose
FRACTION=1

if len(sys.argv) > 3:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
	prefix = sys.argv[3]
else:
        input_dir = '/home/hack17/cd/train/'
        output_dir = '/home/hack17/joe/lmdb/'
	prefix="dogcat"

if len(sys.argv) > 4:
	SPLIT=float(sys.argv[4])

if len(sys.argv) > 5:
	FRACTION=float(sys.argv[5])

if (not cmd_exists("convert_imageset") or not cmd_exists("compute_image_mean")):
	print("Make sure caffe tools are in PATH")
	sys.exit()

if SPLIT > 1.0:
	print("SPLIT can not be greater than 1.0")
	sys.exit()

if FRACTION > 1.0:
        print("FRACTION can not be greater than 1.0")
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

# File extensions to look for
extns = [ '/*.jpeg' , '/*.jpg', '/*.png']
for root, subFolders, files in os.walk(input_dir):
    for extn in extns:
        filename=root + extn
        full_input_path = filename
        addrs = addrs + glob.glob(full_input_path)

print("No. of samples selected for Train is {} out of {}".format(int(SPLIT*FRACTION*len(addrs)), len(addrs)))
print("No. of samples selected for Validation is {} out of {}".format(len(addrs)-int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)), len(addrs)))

#ignore the root input directory to look for labels:
ignore_len = len(input_dir)
if (input_dir[-1] != "/"):
    ignore_len = ignore_len + 1

# labels = [0 if 'cat' in addr else 1 for addr in addrs]
labels = [ labelNames.index(lab) for lab in labelNames  for addr in addrs if lab in addr[ignore_len:] ]

shuffle_data = True
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)


train_addrs = addrs[:int(SPLIT*FRACTION*len(addrs))]
train_labels = labels[:int(SPLIT*FRACTION*len(labels))]

#val_addrs = addrs[int((1-FRACTION)*len(addrs)):]
#val_labels = labels[int((1-FRACTION)*len(labels)):]
val_addrs = addrs[int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)):]
val_labels = labels[int(len(addrs)*(1 -FRACTION + FRACTION* SPLIT)):]
print("train_addrs : {} train_labels : {}". format(len(train_addrs), len(train_labels)))
print("val_addrs : {} val_labels : {}". format(len(val_addrs), len(val_labels)))

train_addrs_labels = zip(train_addrs, train_labels)
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

call(['bash', 'create_imagenet.sh', prefix, inputdir, outpdir, traindata, valdata, 'true', str(img_height), str(img_width)])

print("Conversion complete. LMDB files can be found under {} !!".format(outpdir))

call(['bash', 'make_imagenet_mean.sh', prefix, inputdir, outpdir])

print("Mean image file created under {} !!".format(outpdir))


dict = {
    "OUTPUT_DIR" : outpdir,
    "NUM_LABELS" : str(len(labelNames)),
    "PREFIX"     : prefix,
    "HEIGHT"     : str(img_height),
    "WIDTH"      : str(img_width)
} 

def multiple_replace(dict, text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))

  # For each match, look-up corresponding value in dictionary
  return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text) 

path = os.path.dirname(os.path.realpath(__file__))

with open(path+"/train_test.prototxt.template") as text:
    new_text = multiple_replace(dict, text.read())
with open(outpdir+"/train_test.prototxt", "w") as result:
    result.write(new_text)

with open(path+"/solver.prototxt.template") as text:
    new_text = multiple_replace(dict, text.read())
with open(outpdir+"/solver.prototxt", "w") as result:
    result.write(new_text)

with open(path+"/inference.prototxt.template") as text:
    new_text = multiple_replace(dict, text.read())
with open(outpdir+"/inference.prototxt", "w") as result:
    result.write(new_text)


print("train_test.prototxt & solver.prototxt & inference.prototxt files are created under {} !!".format(outpdir))



