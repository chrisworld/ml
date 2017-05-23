#
# convert to tfrecord to hd5
#

import numpy as np
import tensorflow as tf
import readers

# parameters
tfrecords_filename = 'data/train45.tfrecord'
feature_names = ["audio"]
feature_sizes = [128]
num_epochs = 10
batch_size = 1
num_classes = 4716

# reader instance
reader = readers.YT8MFrameFeatureReader(
    feature_names=feature_names, feature_sizes=feature_sizes)

# prepare
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=num_epochs)

# video tuple:
# video_id, features, labels, ones
batch_video_ids, batch_video_matrix, \
batch_labels, batch_frames  = reader.prepare_reader(filename_queue)

# init
init_op = tf.group(tf.global_variables_initializer(),
    tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    ba_vid, ba_vm, ba_la, ba_fr = sess.run(
        [batch_video_ids, batch_video_matrix, batch_labels, batch_frames])
    print "\n-----video data-----"
    print "video id: ", ba_vid.shape, "\n", ba_vid
    print "video matrix: ", ba_vm.shape, "\n", ba_vm
    print "video labels: ", ba_la.shape, "\n", ba_la
    print "video batch frames: ", ba_fr.shape, "\n", ba_fr

    coord.request_stop()
    coord.join(threads)
