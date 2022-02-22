import numpy as np
import os
import gzip
import struct
import array
import matplotlib.pyplot as plt
import matplotlib.image
from urllib.request import urlretrieve


def download(url, filename):
    if not os.path.exists('data'):
        os.makedirs('data')
    out_file = os.path.join('data', filename)
    if not os.path.isfile(out_file):
        urlretrieve(url, out_file)


def mnist():
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    def parse_labels(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, 'rb') as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in ['train-images-idx3-ubyte.gz',
                     'train-labels-idx1-ubyte.gz',
                     't10k-images-idx3-ubyte.gz',
                     't10k-labels-idx1-ubyte.gz']:
        download(base_url + filename, filename)

    train_images = parse_images('data/train-images-idx3-ubyte.gz')
    train_labels = parse_labels('data/train-labels-idx1-ubyte.gz')
    test_images = parse_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = parse_labels('data/t10k-labels-idx1-ubyte.gz')

    return train_images, train_labels, test_images[:1000], test_labels[:1000]


def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = (partial_flatten(train_images) / 255.0 > .5).astype(float)
    test_images = (partial_flatten(test_images) / 255.0 > .5).astype(float)
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]

    return N_data, train_images, train_labels, test_images, test_labels


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(28, 28),
                cmap=matplotlib.cm.binary, vmin=None, vmax=None):
    """Images should be a (N_images x pixels) matrix."""
    N_images = images.shape[0]
    N_rows = np.int32(np.ceil(float(N_images) / ims_per_row))
    pad_value = np.min(images.ravel())
    concat_images = np.full(((digit_dimensions[0] + padding) * N_rows + padding,
                             (digit_dimensions[1] + padding) * ims_per_row + padding), pad_value)
    for i in range(N_images):
        cur_image = np.reshape(images[i, :], digit_dimensions)
        row_ix = i // ims_per_row
        col_ix = i % ims_per_row
        row_start = padding + (padding + digit_dimensions[0]) * row_ix
        col_start = padding + (padding + digit_dimensions[1]) * col_ix
        concat_images[row_start: row_start + digit_dimensions[0],
                      col_start: col_start + digit_dimensions[1]] = cur_image
        cax = ax.matshow(concat_images, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    return cax


def save_images(images, filename, **kwargs):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111)
    plot_images(images, ax, **kwargs)
    fig.patch.set_visible(False)
    ax.patch.set_visible(False)
    plt.savefig(filename)


def train_mle_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MLE estimators theta_mle and pi_mle"""
    N = len(train_images)
    theta_mle = np.dot(np.transpose(train_images),
                       np.transpose(np.dot(np.diag(np.reciprocal(np.dot(np.transpose(train_labels),
                                                                        np.ones(N)))),
                                           np.transpose(train_labels))))
    pi_mle = np.dot(np.diag(np.reciprocal(N*np.ones(10))), np.dot(np.transpose(train_labels), np.ones(N)))
    return theta_mle, pi_mle


def train_map_estimator(train_images, train_labels):
    """ Inputs: train_images, train_labels
        Returns the MAP estimators theta_map and pi_map"""
    N = len(train_images)
    num = np.add(np.dot(np.transpose(train_images), train_labels), 2*np.ones((784, 10)))
    denom = np.dot(np.reshape(np.ones(784), (-1, 1)), np.transpose(np.reshape(np.reciprocal(np.add(np.dot(np.transpose(train_labels), np.ones(N)), 4*np.ones(10))), (-1, 1))))
    theta_map = np.multiply(num, denom)
    pi_map = np.dot(np.diag(np.reciprocal(N * np.ones(10))), np.dot(np.transpose(train_labels), np.ones(N)))
    return theta_map, pi_map


def log_likelihood(images, theta, pi):
    """ Inputs: images, theta, pi
        Returns the matrix 'log_like' of loglikehoods over the input images where
    log_like[i,c] = log p (c |x^(i), theta, pi) using the estimators theta and pi.
    log_like is a matrix of num of images x num of classes
    Note that log likelihood is not only for c^(i), it is for all possible c's."""
    # Log-likelihood matrix computation omitting the denominator p(x)
    log_pi_mat = np.log(np.dot(np.reshape(np.ones(len(images)), (-1, 1)), np.transpose(np.reshape(pi, (-1, 1)))))
    one_minus_theta = np.subtract(np.ones(theta.shape), theta)
    A = log_prob_x_given_c_theta_pi(np.reshape(images[0], (-1, 1)), theta, one_minus_theta)
    for i in range(1, len(images)):
        prob = log_prob_x_given_c_theta_pi(np.reshape(images[i], (-1, 1)), theta, one_minus_theta)
        A = np.concatenate((A, prob), axis=1)
    A = np.transpose(A)
    log_like_x = np.add(log_pi_mat, A)
    log_sums_x = np.dot(np.reshape(np.log(np.sum(np.exp(log_like_x), axis=1)), (-1, 1)), np.transpose(np.reshape(np.ones(10), (-1, 1))))
    log_like_class = np.subtract(log_like_x, log_sums_x)
    return log_like_class


def log_prob_x_given_c_theta_pi(image, theta, one_minus_theta):
    """ Inputs: image, theta, one_minus_theta
        Returns a 10x1 matrix of the log probability of x for each class c
        given c, theta and pi
    """
    X = np.dot(image, np.ones((1, 10)))
    one_minus_X = np.subtract(np.ones(X.shape), X)
    prob = np.sum(np.add(np.multiply(np.log(theta), X), np.multiply(np.log(one_minus_theta), one_minus_X)), axis=0)
    prob = np.reshape(prob, (-1, 1))
    return prob


def predict(log_like):
    """ Inputs: matrix of log likelihoods
    Returns the predictions based on log likelihood values"""

    predictions = np.argmax(log_like, axis=1)
    return predictions


def accuracy(log_like, labels):
    """ Inputs: matrix of log likelihoods and 1-of-K labels
    Returns the accuracy based on predictions from log likelihood values"""
    predictions = predict(log_like)
    num = 0
    classes = np.argmax(labels, axis=1)
    for i in range(len(predictions)):
        if predictions[i] == classes[i]:
            num = num + 1
    acc = num/len(predictions)
    return acc


def main():
    N_data, train_images, train_labels, test_images, test_labels = load_mnist()

    # Fit MLE and MAP estimators
    theta_mle, pi_mle = train_mle_estimator(train_images, train_labels)
    theta_map, pi_map = train_map_estimator(train_images, train_labels)

    # Find the log likelihood of each data point
    loglike_train_mle = log_likelihood(train_images, theta_mle, pi_mle)
    loglike_train_map = log_likelihood(train_images, theta_map, pi_map)

    avg_loglike_mle = np.sum(loglike_train_mle * train_labels) / N_data
    avg_loglike_map = np.sum(loglike_train_map * train_labels) / N_data

    print("Average log-likelihood for MLE is ", avg_loglike_mle)
    print("Average log-likelihood for MAP is ", avg_loglike_map)

    train_accuracy_map = accuracy(loglike_train_map, train_labels)
    loglike_test_map = log_likelihood(test_images, theta_map, pi_map)
    test_accuracy_map = accuracy(loglike_test_map, test_labels)

    print("Training accuracy for MAP is ", train_accuracy_map)
    print("Test accuracy for MAP is ", test_accuracy_map)

    # Plot MLE and MAP estimators
    save_images(theta_mle.T, 'mle.png')
    save_images(theta_map.T, 'map.png')


if __name__ == '__main__':
    main()
