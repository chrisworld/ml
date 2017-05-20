from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

# parameters
img_path = 'data/Chris-500x333.jpg'
tfrecords_filename = 'data/Chris-500x333.tfrecord'
#img_path = io.imread('data/Chris-500x333.jpg')
IMAGE_HEIGHT = 333
IMAGE_WIDTH = 500

# Convert a image to tfrecord
def write_img2tfrecord(img_path, tfrecords_filename):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    img = np.array(io.imread(img_path))
    height = img.shape[0]
    width = img.shape[1]
    img_raw = img.tostring()
    example = tf.train.Example(
        features = tf.train.Features(
            feature = {
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'image_raw': _bytes_feature(img_raw)}))
    writer.write(example.SerializeToString())
    writer.close()

# Read tfrecord file
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # feature extractor
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    # shaping
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)
    image_size_const = tf.constant(
        (IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    return image

# main
#
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs = 1)
image = read_and_decode(filename_queue)
# init
init_op = tf.group(tf.global_variables_initializer(),
    tf.local_variables_initializer())

print "Read Image: \n"
with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord = coord)

    img = sess.run([image])
    print img[0].shape

    coord.request_stop()
    coord.join(threads)
















# EOF
