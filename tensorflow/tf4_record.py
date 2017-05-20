import tensorflow as tf

# parameters
tfrecords_filename = 'test2S.tfrecord'
num_epochs = 1
num_classes = 4716
feature_sizes = [128]
feature_names = ["mean_audio"]
batch_size = 1024

# session for evaluation
sess = tf.Session()

# FIFO queue
filename_queue = tf.train.string_input_producer(
[tfrecords_filename], num_epochs=num_epochs)

def read_video(filename_queue):
	# Read tfrecord
	reader = tf.TFRecordReader()
	_, serialized_examples = reader.read_up_to(filename_queue, batch_size)

	num_features = len(feature_names)
	feature_map = {
		"video_id": tf.FixedLenFeature([], tf.string),
		"labels": tf.VarLenFeature(tf.int64)}

	for feature_index in range(num_features):
		feature_map[feature_names[feature_index]] = tf.FixedLenFeature(
			[feature_sizes[feature_index]], tf.float32)

	print feature_map
	features = tf.parse_example(serialized_examples, features=feature_map)
	labels = tf.sparse_to_indicator(features["labels"], num_classes)
	labels.set_shape([None, num_classes])
	concatenated_features = tf.concat([
		features[feature_name] for feature_name in feature_names], 1)

	return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])


video = read_video(filename_queue)
print "video: \n", video, "\n"

# init
init_op = tf.group(tf.global_variables_initializer(),
    tf.local_variables_initializer())

with tf.Session() as sess:
	sess.run(init_op)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord = coord)

	vid = sess.run([video])
	#print vid

	coord.request_stop()
	coord.join(threads)


#print "Context: ", contexts['labels'], '\n'
#print "Video_id: ", contexts['video_id'], '\n'

# read ground truth labels
#labels = tf.cast(contexts['labels'], tf.int32)
#print "Labels: ", sess.run(contexts['video_id']), '\n'













# EOF
