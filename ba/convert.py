#
# convert to tfrecord to hd5
#

import numpy as np
import tensorflow as tf
import readers

# parameters
tfrecords_filename = 'data/traineA.tfrecord'
feature_names = ["mean_audio"]
feature_sizes = [128]
num_epochs = 10
batch_size = 10

# reader instance
reader = readers.YT8MAggregatedFeatureReader(
    feature_names=feature_names, feature_sizes=feature_sizes)

# prepare
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=num_epochs)

# video tuple:
# video_id, features, labels, ones
video_id, feat, labels, batch_ones  = reader.prepare_reader(
    filename_queue, batch_size=batch_size)

# init
init_op = tf.group(tf.global_variables_initializer(),
    tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)
    v_id, fea, lab, bat = sess.run([video_id, feat, labels, batch_ones])
    print "\n-----video data-----"
    print "video id: ", v_id.shape, "\n", v_id
    print "video features: ", fea.shape, "\n", fea
    print "video labels: ", lab.shape, "\n", lab
    print "video batch ones: ", bat.shape, "\n", bat
    print "\n-----Some Calculations-----"
    print np.dot(fea, np.ones(fea.shape[1]))

    #for row in vid[0]:
    #	for col in row:
    #		if col: print col

    coord.request_stop()
    coord.join(threads)
