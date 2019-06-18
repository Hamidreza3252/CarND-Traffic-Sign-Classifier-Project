import cv2
import numpy as np
import glob
from sklearn import preprocessing
import tensorflow as tf
import re
import os
from datetime import datetime
import sys
import shutil
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import math
from sklearn import metrics


class Cnn:

    def __init__(self):
        # self.n_classes = 0

        self.label_names = {}
        self.label_ids = []
        self.lb = None

    def init_model(self, label_names, label_ids):
        # self.n_classes = len(label_names)
        # self.label_names = label_names[:]
        self.label_ids = label_ids[:]

        self.label_names = {}
        for label_name, label_id in zip(label_names, label_ids):
            self.label_names[label_id] = label_name

        self.lb = preprocessing.LabelBinarizer()
        self.lb.fit(range(0, len(label_ids)))

    def resize_img(file_name, new_file_name, new_size):
        img = cv2.imread(file_name)
        resized_img = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)

    def get_list_of_image_files(imgs_dir):
        # return glob.glob('SampleRaw/*.txt')
        # return glob.glob('SampleRaw/Lift A - CH12 Data - Breakdown/*.txt')
        return glob.glob(imgs_dir)

    def read_and_resize_images(file_count=-1, skip_file_count=0, images_dir="", specific_file_name="", target_img_size=128, keep_aspect_ratio=True, **kwargs):
        """Read the raw data in a file or all files in the raw data directory and estimate velocity, considering all missing data

        returns:
            a list containing all image objects created
        """

        images = []

        if (file_count == 0):
            return images

        if (specific_file_name == ""):
            list_of_image_files = CnnModules.get_list_of_image_files(images_dir + "*.jpg")
            # list_of_image_files = sorted(list_of_image_files,
            #                            key=lambda x: Lift_Utils.extract_lift_name_and_timestamp(x)[1])

            # list_of_image_files = [x for x in list_of_image_files if x != ""]
        else:
            list_of_image_files = [specific_file_name]

        save_resized_iamges = kwargs["save_resized_iamges"] if("save_resized_iamges" in kwargs) else False
        resized_iamges_dir = kwargs["resized_iamges_dir"] if(save_resized_iamges) else ""

        # print("Reading {} image files...".format(len(list_of_image_files) if (file_count < 0) else min(len(list_of_image_files), file_count)))

        if (file_count != -1):
            file_count += skip_file_count

        for i, file_name in enumerate(list_of_image_files[skip_file_count:], start=skip_file_count):
            # file_name = 'SampleRaw\CH11-(2018, 7, 19, 4, 8, 43, 44, 196).txt'

            # if(i < 5 or i > 5):
            #    continue

            directory, file_name_only = os.path.split(file_name)
            file_name_only = file_name_only[:-4]

            # print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
            # print("{}) {}".format(i, file_name))

            try:
                img = cv2.imread(file_name)

                if(keep_aspect_ratio):
                    img_dimensions = np.asarray(img.shape[:2])

                    target_img_dimensions = np.asarray([target_img_size, target_img_size])
                    img_resize_ratios = target_img_dimensions / img_dimensions
                    min_img_ratio = min(np.min(img_resize_ratios), 1.0)

                    new_img = np.zeros(shape=(target_img_size, target_img_size, 3))

                    # if(np.any(img_resize_ratios < 1.0)):
                    resized_img_dimensions = (img_dimensions * min_img_ratio + 0.5).astype(int)

                    resized_img = cv2.resize(img, dsize=(resized_img_dimensions[1], resized_img_dimensions[0]), interpolation=cv2.INTER_CUBIC)
                    width_gap = int((target_img_size - resized_img_dimensions[0]) // 2.0)
                    height_gap = int((target_img_size - resized_img_dimensions[1]) // 2.0)
                    new_img[width_gap:width_gap + resized_img_dimensions[0], height_gap:height_gap + resized_img_dimensions[1]] = resized_img
                else:
                    new_img = cv2.resize(img, dsize=(target_img_size, target_img_size), interpolation=cv2.INTER_CUBIC)

                if(save_resized_iamges):
                    if(resized_iamges_dir != ""):
                        directory = resized_iamges_dir

                    resized_file_name = directory + file_name_only + "-resized.jpg"
                    cv2.imwrite(resized_file_name, new_img)

                images.append(new_img)
            except Exception as exception:
                print("An error occurred while processing image file {0:} and it was ignored.".format(file_name))

            if (i == file_count - 1):
                break

        return images

    #@staticmethod
    def display_images(self, features, label_ids, normalized=True):

        max_count = 50

        assert (len(features) == len(label_ids)), "len(features) <> len(labels)"
        assert (len(features) < max_count and len(label_ids) < max_count), "len of features must be < {0:d}".format(max_count)

        # label_binarizer = LabelBinarizer()
        # label_binarizer.fit(range(self.n_classes))
        # label_ids = label_binarizer.inverse_transform(np.array(labels))

        # print("Label IDs:", label_ids)

        fig, axes = plt.subplots(nrows=len(features), ncols=1)
        fig.set_figheight(10 + len(features))
        fig.set_figwidth(10)
        fig.tight_layout()
        fig.suptitle("Display sample images", fontsize=10, y=1.02)

        for image_i, (feature, label_id) in enumerate(zip(features, label_ids)):
            label_name = self.label_names[label_id]
            axis = axes[image_i]

            # rectified_feature = (feature * 255.0).astype(int)

            if(normalized):
                axis.imshow((feature * 255).astype(np.uint8))
            else:
                axis.imshow(feature)

            axis.set_title(str(image_i) + ": " + label_name)
            axis.set_axis_off()

    @staticmethod
    def batch_features_labels(features, labels, batch_size):
        """
        Split features and labels into batches
        """

        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield np.asarray(features[start:end]), np.asarray(labels[start:end])

    @staticmethod
    def normalize(x, output_range_min=0.0, output_range_max=1.0, image_data_min=0.0, image_data_max=255.0):
        """
        Normalize a list of sample image data in the range of 0 to 1
        : x: List of image data.  The image shape is (32, 32, 3)
        : return: Numpy array of normalize data
        """

        # output_range_min = 0.0
        # output_range_max = 1.0
        output_range_diff = output_range_max - output_range_min
        # image_data_min = 0.0
        # image_data_max = 255
        image_data_range_diff = image_data_max - image_data_min

        normalized_image_data = output_range_min + (x - image_data_min) * output_range_diff / image_data_range_diff

        return normalized_image_data

    def one_hot_encode(self, x):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """
        return self.lb.transform(x)

    @staticmethod
    def one_hot_encode_2d(x, label_binarizer):
        """
        One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
        : x: List of sample Labels
        : return: Numpy array of one-hot encoded labels
        """

        return np.asarray([[1, 0] if(x_i == 0) else [0, 1] for x_i in x])

    @staticmethod
    def neural_net_image_input(image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images
        : return: Tensor for image input.
        """
        x = tf.placeholder(dtype=tf.float32, shape=[None, image_shape[0], image_shape[1], image_shape[2]], name="x")

        return x

    @staticmethod
    def neural_net_label_input(n_classes):
        """
        Return a Tensor for a batch of label input
        : n_classes: Number of classes
        : return: Tensor for label input.
        """
        y = tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="y")

        return y

    @staticmethod
    def neural_net_keep_prob_input():
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """

        keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")

        return keep_prob

    @staticmethod
    def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides, wieghts_name="", layer_name=""):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """

        # conv_layer = tf.nn.conv2d(input, weight, strides, padding)

        print("conv2d_maxpool... Start")
        print("Checking inputs dimensions...")
        print("conv_ksize:", conv_ksize)
        print("conv_num_outputs:", conv_num_outputs)
        # print(x_tensor)

        input_depth = x_tensor.get_shape().as_list()[3]

        # weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
        # bias = tf.Variable(tf.zeros(k_output))
        # [batch, height, width, channels]

        # truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)

        weights = tf.Variable(tf.truncated_normal(shape=[conv_ksize[0], conv_ksize[1], input_depth, conv_num_outputs], mean=0.0, stddev=0.05), name=wieghts_name)
        biases = tf.Variable(tf.zeros(conv_num_outputs))
        conv_strides = (1, conv_strides[0], conv_strides[1], 1)
        pool_ksize = (1, pool_ksize[0], pool_ksize[1], 1)
        pool_strides = (1, pool_strides[0], pool_strides[1], 1)

        print("Checking strides dimensions...")
        print("conv_strides:", conv_strides)
        print("pool_ksize:", pool_ksize)
        print("pool_strides", pool_strides)

        conv_layer = tf.nn.conv2d(x_tensor, weights, conv_strides, "SAME")
        conv_layer = tf.nn.bias_add(conv_layer, biases, name=layer_name)
        conv_layer = tf.nn.max_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding="SAME")
        conv_layer = tf.nn.relu(conv_layer)

        # H1: conv_layer = tf.nn.max_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding='SAME')

        print("conv2d_maxpool... End")
        print("")

        return conv_layer

    @staticmethod
    def conv2d_avg_pool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """

        # conv_layer = tf.nn.conv2d(input, weight, strides, padding)

        print("conv2d_avg_pool... Start")
        print("Cheking inputs dimensions... ")
        print('conv_ksize: ', conv_ksize)
        print('conv_num_outputs: ', conv_num_outputs)
        # print(x_tensor)

        input_depth = x_tensor.get_shape().as_list()[3]

        # weight = tf.Variable(tf.truncated_normal([filter_size_height, filter_size_width, color_channels, k_output]))
        # bias = tf.Variable(tf.zeros(k_output))
        # [batch, height, width, channels]

        """
        truncated_normal(
        shape,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=None,
        name=None
        )
        """

        weights = tf.Variable(tf.truncated_normal(shape=[conv_ksize[0], conv_ksize[1], input_depth, conv_num_outputs], mean=0.0, stddev=0.05))
        biases = tf.Variable(tf.zeros(conv_num_outputs))
        conv_strides = (1, conv_strides[0], conv_strides[1], 1)
        pool_ksize = (1, pool_ksize[0], pool_ksize[1], 1)
        pool_strides = (1, pool_strides[0], pool_strides[1], 1)

        print("Cheking strides dimensions... ")
        print('conv_strides: ', conv_strides)
        print('pool_ksize: ', pool_ksize)
        print('pool_strides', pool_strides)

        conv_layer = tf.nn.conv2d(x_tensor, weights, conv_strides, "SAME")
        conv_layer = tf.nn.bias_add(conv_layer, biases)
        conv_layer = tf.nn.avg_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding="SAME")
        conv_layer = tf.nn.relu(conv_layer)

        # H1: conv_layer = tf.nn.max_pool(conv_layer, ksize=pool_ksize, strides=pool_strides, padding='SAME')

        print("conv2d_avg_pool... End")
        print("")
        return conv_layer

    @staticmethod
    def flatten(x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """

        # print(x_tensor)

        output_tensor = tf.contrib.layers.flatten(x_tensor)

        # print(output_tensor)

        return output_tensor

    @staticmethod
    def fully_conn(x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """

        """
        fully_connected(
        inputs,
        num_outputs,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=initializers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
        )
        """

        output_tensor = tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=tf.nn.relu)

        return output_tensor

    @staticmethod
    def output(x_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """

        output_tensor = tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=None)
        # output_tensor = tf.contrib.layers.fully_connected(x_tensor, num_outputs, activation_fn=tf.nn.sigmoid)

        return output_tensor

    @staticmethod
    def train_neural_network(session, x_tf_ph, y_tf_ph, keep_prob_tf_ph, optimizer, keep_probability, feature_batch, label_batch):
        """
        Optimize the session on a batch of images and labels
        : session: Current TensorFlow session
        : optimizer: TensorFlow optimizer function
        : keep_probability: keep probability
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        """
        # batch_size.shape  ->  (128, 32, 32, 3)
        # label_batch.shape  ->  (128, 10)

        session.run(optimizer, feed_dict={x_tf_ph: feature_batch, y_tf_ph: label_batch, keep_prob_tf_ph: keep_probability})

    @staticmethod
    def print_stats(session, x_tf_ph, y_tf_ph, keep_prob_tf_ph, feature_batch, label_batch, val_images, val_labels, cost, accuracy):
        """
        Print information about loss and validation accuracy
        : session: Current TensorFlow session
        : feature_batch: Batch of Numpy image data
        : label_batch: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """

        # print(cost)
        # print(accuracy)

        # correct_prediction = tf.equal(tf.argmax(valid_labels, 1), tf.argmax(label_batch, 1))

        test_cost = session.run(cost, feed_dict={x_tf_ph: feature_batch, y_tf_ph: label_batch, keep_prob_tf_ph: 1.0})
        valid_accuracy = session.run(accuracy, feed_dict={x_tf_ph: val_images, y_tf_ph: val_labels, keep_prob_tf_ph: 1.0})

        print("Test Cost: {0:0.4f}   ---   Valid Accuracy: {1:0.4f}".format(test_cost, valid_accuracy))
        # print('Test Accuracy: {}'.format(test_accuracy))

    @staticmethod
    def _load_label_names():
        """
        Load the label names from file
        """
        raise NotImplementedError("deprecated method")

    @staticmethod
    def display_image_predictions(features, labels, predictions, n_classes):
        label_names = CnnModules._load_label_names()
        label_binarizer = LabelBinarizer()
        label_binarizer.fit(range(n_classes))
        label_ids = label_binarizer.inverse_transform(np.array(labels))

        fig, axies = plt.subplots(nrows=len(features), ncols=2)
        fig.set_figheight(12)
        fig.set_figwidth(12)
        fig.tight_layout()
        fig.suptitle("Softmax Predictions", fontsize=20, y=1.1)

        n_predictions = n_classes
        margin = 0.05
        ind = np.arange(n_predictions)
        width = (1. - 2. * margin) / n_predictions

        for image_i, (feature, label_id, pred_indicies, pred_values) in enumerate(zip(features, label_ids, predictions.indices, predictions.values)):
            pred_names = [label_names[pred_i] for pred_i in pred_indicies]
            correct_name = label_names[label_id]

            # rectified_feature = (feature * 255.0).astype(int)

            axies[image_i][0].imshow((feature * 255).astype(np.uint8))
            axies[image_i][0].set_title(correct_name)
            axies[image_i][0].set_axis_off()

            axies[image_i][1].barh(ind + margin, pred_values[::-1], width)
            axies[image_i][1].set_yticks(ind + margin)
            axies[image_i][1].set_yticklabels(pred_names[::-1])
            axies[image_i][1].set_xticks([0, 0.5, 1.0])

    @staticmethod
    def evaluate_roc_curve(test_labels, predicted_y_probabilities):

        fpr, tpr, _ = metrics.roc_curve(test_labels, predicted_y_probabilities)
        auc = metrics.roc_auc_score(test_labels, predicted_y_probabilities)

        return fpr, tpr, auc

    @staticmethod
    def plot_roc_curve(test_labels, predicted_y_probabilities, **kwargs):

        title = kwargs["title"] if("title" in kwargs) else ""
        legend_title = kwargs["legend_title"] if("legend_title" in kwargs) else "data-1, auc"

        fpr, tpr, _ = metrics.roc_curve(test_labels, predicted_y_probabilities)
        auc = metrics.roc_auc_score(test_labels, predicted_y_probabilities)

        fig, axis = plt.subplots()

        axis.plot(fpr, tpr, label=legend_title + "={0:0.5f}".format(auc))
        axis.legend(loc=4, fontsize="x-large")
        axis.set_title(title, fontsize="x-large")

        gridlines = axis.get_xgridlines() + axis.get_ygridlines()
        ticklabels = axis.get_xticklabels() + axis.get_yticklabels()

        for line in gridlines:
            line.set_linestyle('-.')

        for label in ticklabels:
            label.set_color("b")
            label.set_fontsize(13)

        axis.grid(b=True, which="both", alpha=1.0)

        fig.set_figheight(6)
        fig.set_figwidth(10)
        fig.tight_layout()

        plt.show()

        return fpr, tpr, auc

    @staticmethod
    def plot_evaluated_roc_curves(fprs, tprs, acus, legend_titles, **kwargs):

        title = kwargs["title"] if("title" in kwargs) else ""

        if(len(legend_titles) != len(fprs)):
            raise Exception("list lengths mismatch")

        fig, axis = plt.subplots()

        for i, fpr, tpr, auc, legend_title in zip(range(len(fprs)), fprs, tprs, acus, legend_titles):
            axis.plot(fpr, tpr, label=legend_title + "={0:0.5f}".format(auc))
            axis.legend(loc=4, fontsize="x-large")

        axis.set_title(title, fontsize="x-large")

        gridlines = axis.get_xgridlines() + axis.get_ygridlines()
        ticklabels = axis.get_xticklabels() + axis.get_yticklabels()

        for line in gridlines:
            line.set_linestyle('-.')

        for label in ticklabels:
            label.set_color("b")
            label.set_fontsize(13)

        axis.grid(b=True, which="both", alpha=1.0)

        fig.set_figheight(6)
        fig.set_figwidth(10)
        fig.tight_layout()

        plt.show()

    @staticmethod
    def get_activations(sess, image_in_tf, keep_prob_tf, layer, stimuli):
        # units = sess.run(layer, feed_dict={image_in_tf:np.reshape(stimuli, [1, -1], order="F"), keep_prob_tf:1.0})
        units = sess.run(layer, feed_dict={image_in_tf: np.reshape(stimuli, [-1, stimuli.shape[0], stimuli.shape[1], stimuli.shape[2]]), keep_prob_tf: 1.0})
        # units = sess.run(layer, feed_dict={image_in_tf:stimuli, keep_prob_tf:1.0})
        # print(units.shape)
        return units

    @staticmethod
    def plot_nn_filter(units, n_columns):
        filter_count = units.shape[3]
        plt.figure(1, figsize=(20, 20))
        # n_columns = 8
        n_rows = math.ceil(filter_count / n_columns) + 1

        for i in range(filter_count):
            axis = plt.subplot(n_rows, n_columns, i + 1)

            plt.title("Filter " + str(i + 1))
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")

            axis.set_axis_off()

    @staticmethod
    def load_preprocess_training_batch(batch_id, batch_size):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        filename = 'preprocess_batch_' + str(batch_id) + '.p'
        features, labels = pickle.load(open(filename, mode='rb'))

        # Return the training data in batches of size <batch_size> or less
        return Cnn.batch_features_labels(features, labels, batch_size)
