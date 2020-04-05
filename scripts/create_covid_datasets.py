from random import shuffle
import glob
import sys
import cv2
import numpy as np
#import skimage.io as io
import tensorflow as tf
from absl import app
from tqdm import trange

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _encode_png(images):
#     raw = []
#     with tf.compat.v1.Session() as sess, tf.device('cpu:0'):
#         image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
#         to_png = tf.image.encode_png(image_x)
#         for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
#             raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
#     return raw

def load_image(addr):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    img = cv2.imread(addr)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) #convert gray images to rgb

    return img
 
def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 10:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        img = load_image(addrs[i])

        label = labels[i]

        if img is None:
            continue

        # Create a feature
        # feature = {
        #     'image': _bytes_feature(img.tostring()),
        #     'label': _int64_feature(label)
        # }
        feat = dict(image=_bytes_feature(img.tostring()),
                    label=_int64_feature(label))
        # Create an example protocol buffer
        record = tf.train.Example(features=tf.train.Features(feature=feat))
        
        # Serialize to string and write on the file
        writer.write(record.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

# cat_dog_train_path = 'PetImages/*/*.jpg'
labelled_path = './Labelled_Images/*/*.*g'
# read addresses and labels from the 'train' folder
# addrs = glob.glob(cat_dog_train_path)
addrs = glob.glob(labelled_path)
# labels = [0 if 'Cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
labels = [\
        1 if 'covid' in addr in addrs \
        2 if 'bacterial_pneumonia' in addr in addrs \
        3 if 'viral_pneumonia' in addr in addrs \
        4 if 'normal' in addr in addrs \
        ]  # 0 = Cat, 1 = Dog

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)
    
# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.8*len(addrs))]
train_labels = labels[0:int(0.8*len(labels))]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

createDataRecord('covid-train.tfrecord', train_addrs, train_labels)
# createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('covid-test.tfrecord', test_addrs, test_labels)


#-----------For unlabelled dataset creation-----------------------
# cat_dog_train_path = 'PetImages/*/*.jpg'
unlabelled_path = './Unlabelled_Images/*/*.*g'
# read addresses and labels from the 'train' folder
# addrs = glob.glob(cat_dog_train_path)
addrs = glob.glob(unlabelled_path)
# labels = [0 if 'Cat' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
labels = [\
        0 if 'unlabelled' in addr in addrs \
        # 0 if 'bacterial_pneumonia' in addr in addrs \
        # 0 if 'viral_pneumonia' in addr in addrs \
        # 0 if 'normal' in addr in addrs \
        ]  # 0 = Cat, 1 = Dog

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)
    
# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:]
train_labels = labels[0:]
# val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
# val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
# test_addrs = addrs[int(0.8*len(addrs)):]
# test_labels = labels[int(0.8*len(labels)):]

createDataRecord('covid-unlabel.tfrecord', train_addrs, train_labels)
# createDataRecord('val.tfrecords', val_addrs, val_labels)
# createDataRecord('test.tfrecords', test_addrs, test_labels)