# isdlhackathon

git clone https://github.com/josiahsams/isdlhackathon

cd isdlhackathon

### Prereqs:

To enable Tensorflow framework in the environment you need to run the following command after installing Tensorflow:
```
$ source /opt/DL/tensorflow/bin/tensorflow-activate
```
To enable Caffe framework in the environment you need to run the following command after installing Caffe:
```
$ source /opt/DL/caffe-ibm/bin/caffe-activate
```

The input_directory should have the image files extracted and a file named "labels.txt". The "labels.txt" should 
list all the predefined labels the dataset is classified in the newline separated style as follows,

```
cat <input_directory>/labels.txt
cats
dogs
```

### To convert a directory of images(jpg) into TensorflowRecords,

``` ./tfrecord/convert2TFrecordsv2.py <input_directory> <output_directory> <split> <fraction>```

* split & fraction are optional parameters.
* Split denotes the train & validation data split (default 0.5 - 50& train & 50% validation data split)
* If split is .75 then train data is 75% and validation data is 20% of the entire dataset.
* Fraction is used for experimentation and indicated the amount of total dataset to be processed (default 1 - 100% data is processed)
* both should be less or equal to 1

On completion, the `output directory` will have files as follows,
```
hack17@fsmldlpar7:~/joe$ ls -lt /home/hack17/joe/tfrecord/
total 3464
-rw-rw-r-- 1 hack17 hack17 1825135 Jul 29 05:28 validation-00000-of-00001
-rw-rw-r-- 1 hack17 hack17 1718344 Jul 29 05:27 train-00000-of-00001
```

To train using `inception model in TensorFlow`, provide the `output directory` to the command as follows,
```
cd tensorflow-models/inception
bazel build //inception:imagenet_train

# run it
bazel-bin/inception/imagenet_train 
  --num_gpus=2 
  --batch_size=64 
  --train_dir=/tmp/mydata1 
  --data_dir=/home/hack17/joe/tfrecord
```
Reference [link](https://github.com/tensorflow/models/tree/master/inception)

### To convert a directory of images(jpg) into LMDB(caffe),

``` ./lmdb/convert2LMDBv2.py <input_directory> <output_directory> <prefix> <split> <fraction>```

* split & fraction are optional parameters.
* Split denotes the train & validation data split (default 0.5 - 50& train & 50% validation data split)
* If split is .75 then train data is 75% and validation data is 20% of the entire dataset.
* Fraction is used for experimentation and indicated the amount of total dataset to be processed (default 1 - 100% data is processed)
* both should be less or equal to 1

On completion, the output directory will have files as follows,
```
hack17@fsmldlpar7:~/joe$ ls -lt /home/hack17/joe/lmdb/
total 800
-rw-rw-r-- 1 hack17 hack17    296 Jul 29 05:25 solver.prototxt
-rw-rw-r-- 1 hack17 hack17   5370 Jul 29 05:25 train_val.prototxt
-rw-rw-r-- 1 hack17 hack17 786446 Jul 29 05:25 dogcat_mean.binaryproto
drwxr--r-- 2 hack17 hack17   4096 Jul 29 05:25 dogcat_val_lmdb
drwxr--r-- 2 hack17 hack17   4096 Jul 29 05:25 dogcat_train_lmdb
-rw-rw-r-- 1 hack17 hack17   1699 Jul 29 05:25 test.txt
-rw-rw-r-- 1 hack17 hack17   1596 Jul 29 05:25 train.txt
```

To train using `Alexnet model in Caffe`, provide the output directory to the command as follows,
```
caffe train --solver=/home/hack17/joe/lmdb/solver.prototxt
```
Reference [link](http://caffe.berkeleyvision.org/gathered/examples/imagenet.html)

### To convert a directory of images(jpg) into Raw format,

``` ./raw/convert2Raw.py <input_directory> <output_directory> <split> <fraction>```

* split & fraction are optional parameters.
* Split denotes the train & validation data split (default 0.5 - 50& train & 50% validation data split)
* If split is .75 then train data is 75% and validation data is 20% of the entire dataset.
* Fraction is used for experimentation and indicated the amount of total dataset to be processed (default 1 - 100% data is processed)
* both should be less or equal to 1

On completion, 2 files named `train.txt` and `validate.txt` are created in the output directory.
They hold the file path with the label indexes for the files part of training & validation sets.
The content is as follows,

```
hack17@fsmldlpar7:~/joe/repo/raw$ cat /home/hack17/joe/raw/train.txt
dir1/dir2/cat.2976.raw 1
dir1/dir2/cat.3666.raw 1
dir1/cat.8294.raw 0
/dog.3193.raw 0
dir1/cat.965.raw 1
dir1/cat.6546.raw 1
```

Note: For image conversion we use "cv2" python library.

