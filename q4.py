'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''
import matplotlib

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    one_hot = np.zeros((len(train_labels), 10))
    rows = np.arange(train_labels.size)
    for i in range(len(train_labels)):
        one_hot[rows[i], int(train_labels[i])] = 1
    means = np.dot(np.dot(np.diag(np.reciprocal(np.dot(np.transpose(one_hot), np.ones(len(one_hot))))), np.transpose(one_hot)), train_data)
    return means


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    one_hot = np.zeros((len(train_labels), 10))
    rows = np.arange(train_labels.size)
    for i in range(len(train_labels)):
        one_hot[rows[i], int(train_labels[i])] = 1
    mean_mles = compute_mean_mles(train_data, train_labels)
    covariances = np.atleast_3d(get_class_sigma_mle(train_data, one_hot, 0, np.transpose(np.reshape(mean_mles[0], (-1, 1)))))
    for i in range(1, 10):
        sig = get_class_sigma_mle(train_data, one_hot, i, np.transpose(np.reshape(mean_mles[i], (-1, 1))))
        covariances = np.dstack((covariances, sig))
    return np.transpose(covariances)


def get_class_sigma_mle(train_data, one_hot, index, mean_mle):
    class_bool = np.transpose(one_hot)[index]
    train_data_class = train_data[class_bool.astype(bool)]
    N = len(train_data_class)
    sigma_mle = 1/N * np.dot(np.transpose(np.subtract(train_data_class, np.dot(np.reshape(np.ones(N), (-1, 1)), mean_mle))),
                             np.subtract(train_data_class, np.dot(np.reshape(np.ones(N), (-1, 1)), mean_mle)))
    sigma_mle = np.add(sigma_mle, 0.01*np.identity(len(sigma_mle)))
    return sigma_mle


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    log_like = generative_likelihood_per_data(digits[0], means, covariances)
    for i in range(1, len(digits)):
        log_like = np.concatenate((log_like, generative_likelihood_per_data(digits[i], means, covariances)))
    return log_like


def generative_likelihood_per_data(digit, means, covariances):
    '''
    Compute the generative log-likelihood for one data point
    '''
    gen = []
    for i in range(10):
        gen.append(-0.5*np.dot(np.dot(np.transpose(np.subtract(digit, np.transpose(means[i]))),
                                 np.linalg.inv(covariances[i])), np.subtract(digit, np.transpose(means[i]))) -
                   32*np.log(2*np.pi) - 0.5*np.logaddexp(np.linalg.det(covariances[i]), 0))
    return np.transpose(np.reshape(np.array(gen), (-1, 1)))


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    log_like = generative_likelihood(digits, means, covariances)
    log_exp_sums = np.log(np.sum(np.exp(log_like), axis=1))
    cond_like = np.subtract(log_like, np.dot(np.reshape(log_exp_sums, (-1, 1)), np.transpose(np.reshape(np.ones(10), (-1, 1)))))
    return cond_like


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    sum_cor = 0
    for i in range(len(cond_likelihood)):
        sum_cor = sum_cor + cond_likelihood[i][int(labels[i])]
    return sum_cor/len(digits)


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    #gen_likelihood = generative_likelihood(digits, means, covariances)
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    predictions = np.argmax(cond_likelihood, axis=1)
    #predictions = np.argmax(gen_likelihood, axis=1)
    return predictions


def accuracy(predictions, labels):
    cor = 0
    for i in range(len(predictions)):
        if (predictions[i] == int(labels[i])):
            cor = cor + 1
    return cor/len(predictions)


def plot_images(images, ax, ims_per_row=5, padding=5, digit_dimensions=(8, 8),
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


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)
    # Evaluation
    # a)
    print("Average conditional log likelihood of true class labels of the training data: ", avg_conditional_likelihood(train_data, train_labels, means, covariances))
    print("Average conditional log likelihood of true class labels of the test data: ", avg_conditional_likelihood(test_data, test_labels, means, covariances))

    # b)
    training_predictions = classify_data(train_data, means, covariances)
    test_predictions = classify_data(test_data, means, covariances)
    print("Training accuracy: ", accuracy(training_predictions, train_labels))
    print("Test accuracy: ", accuracy(test_predictions, test_labels))

    # c)
    eigenvectors = []
    for i in range(10):
        values, vectors = np.linalg.eig(covariances[i])
        leading = np.argmax(values)
        eigenvectors.append(vectors[:,leading])
    eigenvectors = np.array(eigenvectors)
    save_images(eigenvectors, "eigenvectors.png")


if __name__ == '__main__':
    main()
