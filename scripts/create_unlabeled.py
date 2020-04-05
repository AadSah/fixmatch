#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to create SSL splits from a dataset.
"""

import json
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from absl import app
from tqdm import trange, tqdm

from libml import utils


def get_class(serialized_example):
    return tf.compat.v1.io.parse_single_example(serialized_example, features={'label': tf.compat.v1.io.FixedLenFeature([], tf.int64)})['label']
#Just returns the label of an item

def main(argv):
    argv.pop(0)
    if any(not tf.compat.v1.gfile.Exists(f) for f in argv[1:]):
        raise FileNotFoundError(argv[1:])
    target = argv[0]
    input_files = argv[1:]
    count = 0
    id_class = []
    class_id = defaultdict(list)
    print('Computing class distribution')
    tf.compat.v1.disable_eager_execution() #Added
    dataset = tf.compat.v1.data.TFRecordDataset(input_files).map(get_class, 4).batch(1 << 10)
    it = dataset.make_one_shot_iterator().get_next()
    try:
        with tf.compat.v1.Session() as session, tqdm(leave=False) as t:
            while 1:
                old_count = count
                for i in session.run(it):
                    id_class.append(i) #id_class = [ 1, 0, 4, 3, 8, ...] 
                    class_id[i].append(count)
                    count += 1
                t.update(count - old_count)
    except tf.errors.OutOfRangeError:
        pass
    print('%d records found' % count)
    nclass = len(class_id) #class_id = {0: [] , 1: [], ...}
    assert min(class_id.keys()) == 0 and max(class_id.keys()) == (nclass - 1)
    train_stats = np.array([len(class_id[i]) for i in range(nclass)], np.float64) #Number of Images belonging to each class...
    train_stats /= train_stats.max()
    if 'stl10' in argv[1]:
        # All of the unlabeled data is given label 0, but we know that
        # STL has equally distributed data among the 10 classes.
        train_stats[:] = 1

    print('  Stats', ' '.join(['%.2f' % (100 * x) for x in train_stats]))
    del class_id

    print('Creating unlabeled dataset for in %s' % target)
    npos = np.zeros(nclass, np.int64) #[0, 0, 0, ...]
    class_data = [[] for _ in range(nclass)]
    unlabel = []
    tf.compat.v1.gfile.MakeDirs(os.path.dirname(target))
    with tf.compat.v1.python_io.TFRecordWriter(target + '-unlabel.tfrecord') as writer_unlabel:
        pos, loop = 0, trange(count, desc='Writing records')
        for input_file in input_files: #can be skipped for us...
            for record in tf.compat.v1.python_io.tf_record_iterator(input_file):
                class_data[id_class[pos]].append((pos, record)) #class_data = [ [(1, 1th record)] [] [(0, 0th record)] [] [] ... []]
                while True:
                    c = np.argmax(train_stats - npos / max(npos.max(), 1))
                    if class_data[c]:
                        p, v = class_data[c].pop(0)
                        unlabel.append(p)
                        writer_unlabel.write(v)
                        npos[c] += 1 #npos saves for each class, how many went to unlabelled...
                    else:
                        break
                pos += 1
                loop.update()
        for remain in class_data:
            for p, v in remain:
                unlabel.append(p)
                writer_unlabel.write(v)
        loop.close()
    with tf.compat.v1.gfile.Open(target + '-unlabel.json', 'w') as writer:
        writer.write(json.dumps(dict(distribution=train_stats.tolist(), indexes=unlabel), indent=2, sort_keys=True))


if __name__ == '__main__':
    utils.setup_tf()
    app.run(main)
