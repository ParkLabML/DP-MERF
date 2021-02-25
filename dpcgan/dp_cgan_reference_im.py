"""
copied with minor changes from
https://github.com/reihaneh-torkzadehmahani/DP-CGAN/blob/master/DP_CGAN/dp_conditional_gan_mnist/DP_CGAN_RdpAcc.py
"""

from mlxtend.data import loadlocal_mnist
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize

# Import the requiered python packages
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import argparse

# Import required Differential Privacy packages
baseDir = "../"
sys.path.append(baseDir)

from dp_cgan_accounting_dep.dp_sgd.dp_optimizer import dp_optimizer
from dp_cgan_accounting_dep.dp_sgd.dp_optimizer import sanitizer
from dp_cgan_accounting_dep.dp_sgd.dp_optimizer import utils
from dp_cgan_accounting_dep.privacy_accountant.tf import accountant
from dp_cgan_accounting_dep.analysis.rdp_accountant import compute_rdp
from dp_cgan_accounting_dep.analysis.rdp_accountant import get_privacy_spent




def compute_fpr_tpr_roc(y_test, y_score):
    n_classes = y_score.shape[1]
    false_positive_rate = dict()
    true_positive_rate = dict()
    roc_auc = dict()
    for class_cntr in range(n_classes):
        # independently computes auroc curve for each label and class preds
        #
        false_positive_rate[class_cntr], true_positive_rate[class_cntr], _ = roc_curve(y_test[:, class_cntr],
                                                                                       y_score[:, class_cntr])
        roc_auc[class_cntr] = auc(false_positive_rate[class_cntr], true_positive_rate[class_cntr])

    # Compute micro-average ROC curve and ROC area
    false_positive_rate["micro"], true_positive_rate["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(false_positive_rate["micro"], true_positive_rate["micro"])

    return false_positive_rate, true_positive_rate, roc_auc


def classify(x_train, y_train, y_test, classifer_name, random_state_value=0):
    if classifer_name == "svm":
        classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state_value))
    elif classifer_name == "dt":
        classifier = OneVsRestClassifier(DecisionTreeClassifier(random_state=random_state_value))
    elif classifer_name == "lr":
        classifier = OneVsRestClassifier(
            LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=random_state_value))
    elif classifer_name == "rf":
        classifier = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=random_state_value))
    elif classifer_name == "gnb":
        classifier = OneVsRestClassifier(GaussianNB())
    elif classifer_name == "bnb":
        classifier = OneVsRestClassifier(BernoulliNB(alpha=.01))
    elif classifer_name == "ab":
        classifier = OneVsRestClassifier(AdaBoostClassifier(random_state=random_state_value))
    elif classifer_name == "mlp":
        classifier = OneVsRestClassifier(MLPClassifier(random_state=random_state_value, alpha=1))
    else:
        print("Classifier not in the list!")
        exit()

    y_score = classifier.fit(x_train, y_train).predict_proba(y_test)  # gets class probabilities for each class here
    return y_score


def xavier_init(size):
    """ Xavier Function to keep the scale of the gradients roughly the same
        in all the layers.
    """
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_z(m, n):
    """ Function to generate uniform prior for G(z)
    """
    return np.random.uniform(-1., 1., size=[m, n])


def generator(z, y, theta_g):
    g_w1 = theta_g[0]
    g_w2 = theta_g[1]
    g_b1 = theta_g[2]
    g_b2 = theta_g[3]

    """ Function to build the generator network
    """
    inputs = tf.concat(axis=1, values=[z, y])
    g_h1 = tf.nn.relu(tf.matmul(inputs, g_w1) + g_b1)
    g_log_prob = tf.matmul(g_h1, g_w2) + g_b2
    g_prob = tf.nn.sigmoid(g_log_prob)
    return g_prob


def discriminator(x, y, theta_d):
    """ Function to build the discriminator network
    """
    d_w1 = theta_d[0]
    d_w2 = theta_d[1]
    d_b1 = theta_d[2]
    d_b2 = theta_d[3]

    inputs = tf.concat(axis=1, values=[x, y])
    d_h1 = tf.nn.relu(tf.matmul(inputs, d_w1) + d_b1)
    d_logit = tf.matmul(d_h1, d_w2) + d_b2
    d_prob = tf.nn.sigmoid(d_logit)

    return d_prob, d_logit


def plot(samples):
    """ Function to plot the generated images
    """
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(10, 1)
    gs.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        # plt.show()
    return fig


def del_all_flags(FLAGS):
    """ Function to delete all flags before declare
    """
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def compute_epsilon(batch_size, steps, sigma):
    """Computes epsilon value for given hyperparameters."""
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = batch_size / 60000
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=sigma,
                      steps=steps,
                      orders=orders)
    # Delta is set to 1e-5 because MNIST has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def runTensorFlow(sigma, clipping_value, batch_size, epsilon, delta, iteration, data_save_str,
                  dataset_key, run_auc_test):
    h_dim = 128
    Z_dim = 100

    # Initializations for a two-layer discriminator network
    # code_balanced = input_data.read_data_sets(baseDir + "our_dp_conditional_gan_mnist/mnist_dataset", one_hot=True)
    if dataset_key == 'digits':
        mnist = input_data.read_data_sets("data/MNIST/raw", one_hot=True)
    elif dataset_key == 'fashion':
        print('using FashionMNIST')
        mnist = input_data.read_data_sets("../../data/FashionMNIST/raw", one_hot=True)
    else:
        raise ValueError

    x_dim = mnist.train.images.shape[1]
    y_dim = mnist.train.labels.shape[1]
    x_pl = tf.placeholder(tf.float32, shape=[None, x_dim])
    y_pl = tf.placeholder(tf.float32, shape=[None, y_dim])

    d_w1 = tf.Variable(xavier_init([x_dim + y_dim, h_dim]))
    d_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    d_w2 = tf.Variable(xavier_init([h_dim, 1]))
    d_b2 = tf.Variable(tf.zeros(shape=[1]))

    theta_d = [d_w1, d_w2, d_b1, d_b2]

    # Initializations for a two-layer genrator network
    z_pl = tf.placeholder(tf.float32, shape=[None, Z_dim])
    g_w1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
    g_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    g_w2 = tf.Variable(xavier_init([h_dim, x_dim]))
    g_b2 = tf.Variable(tf.zeros(shape=[x_dim]))
    theta_g = [g_w1, g_w2, g_b1, g_b2]

    start_lr = 0.1
    end_lr = 0.052
    lr_saturate_epochs = 10000
    batches_per_lot = 1
    num_training_steps = 100000
    # num_training_steps = 30000

    # Set accountant type to GaussianMomentsAccountant
    num_training_images = 60000
    priv_accountant = accountant.GaussianMomentsAccountant(num_training_images)

    # Sanitizer
    # batch_size = FLAGS.batch_size
    # clipping_value = FLAGS.default_gradient_l2norm_bound
    # clipping_value = tf.placeholder(tf.float32)
    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(priv_accountant, [clipping_value / batch_size, True])

    # Instantiate the Generator Network
    g_sample = generator(z_pl, y_pl, theta_g)

    # Instantiate the Discriminator Network
    d_real, d_logit_real = discriminator(x_pl, y_pl, theta_d)
    d_fake, d_logit_fake = discriminator(g_sample, y_pl, theta_d)

    # Discriminator loss for real data
    d_loss_real_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_real, labels=tf.ones_like(d_logit_real))
    d_loss_real = tf.reduce_mean(d_loss_real_ce, [0])
    # Discriminator loss for fake data
    d_loss_fake_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.zeros_like(d_logit_fake))
    d_loss_fake = tf.reduce_mean(d_loss_fake_ce, [0])

    # Generator loss
    g_loss_ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logit_fake, labels=tf.ones_like(d_logit_fake))
    g_loss = tf.reduce_mean(g_loss_ce, [0])

    # ------------------------------------------------------------------------------
    """
    minimize_ours :
            Our method (Clipping the gradients of loss on real data and making
            them noisy + Clipping the gradients of loss on fake data) is
            implemented in this function .
            It can be found in the following directory:
            differential_privacy/dp_sgd/dp_optimizer/dp_optimizer.py'
    """
    lr_pl = tf.placeholder(tf.float32)
    # sigma = FLAGS.sigma
    # Generator optimizer
    g_solver = tf.train.AdamOptimizer().minimize(g_loss, var_list=theta_g)
    # Discriminator Optimizer
    d_optim = dp_optimizer.DPGradientDescentOptimizer(lr_pl, [None, None], gaussian_sanitizer, sigma=sigma,
                                                       batches_per_lot=batches_per_lot)
    d_solver = d_optim.minimize_ours(d_loss_real, d_loss_fake, var_list=theta_d)
    # ------------------------------------------------------------------------------

    # Set output directory
    result_dir = baseDir + "out/"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    result_path = result_dir + "/run_{}_bs_{}_s_{}_c_{}_d_{}_e_{}".format(iteration, batch_size, sigma, clipping_value,
                                                                          delta, str(epsilon))

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    target_eps = [float(s) for s in str(epsilon).split(",")]
    max_target_eps = max(target_eps)

    # gpu_options = tf.GPUOptions(visible_device_list="0, 1")
    gpu_options = tf.GPUOptions(visible_device_list="0")

    # Main Session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        init = tf.initialize_all_variables()
        sess.run(init)

        step = 0

        # Is true when the spent privacy budget exceeds the target budget
        should_terminate = False

        # Main loop
        while step <= num_training_steps and should_terminate is False:

            epoch = step
            curr_lr = utils.VaryRate(start_lr, end_lr, lr_saturate_epochs, epoch)

            eps = compute_epsilon(batch_size, (step + 1), sigma * clipping_value)

            # Save the generated images every 50 steps
            if step % 50 == 0:
                print("step :  " + str(step) + "  eps : " + str(eps))

                n_sample = 10
                z_sample = sample_z(n_sample, Z_dim)
                y_sample = np.eye(10)

                samples = sess.run(g_sample, feed_dict={z_pl: z_sample, y_pl: y_sample})

                fig = plot(samples)
                plt.savefig(
                    (result_path + "/step_{}.png").format(str(step).zfill(3)), bbox_inches='tight')
                plt.close(fig)

            x_mb, y_mb = mnist.train.next_batch(batch_size, shuffle=True)

            z_sample = sample_z(batch_size, Z_dim)

            # Update the discriminator network
            _, d_loss_real_curr, d_loss_fake_curr = sess.run([d_solver, d_loss_real, d_loss_fake],
                                                             feed_dict={x_pl: x_mb, z_pl: z_sample, y_pl: y_mb, lr_pl: curr_lr})

            # Update the generator network
            _, g_loss_curr = sess.run([g_solver, g_loss], feed_dict={z_pl: z_sample, y_pl: y_mb, lr_pl: curr_lr})

            if eps >= max_target_eps or step >= num_training_steps:
                print("TERMINATE!!!!")
                print("Termination Step : " + str(step))
                should_terminate = True

                for i in range(0, 10):
                    n_sample = 10
                    z_sample = sample_z(n_sample, Z_dim)

                    y_sample = np.eye(10)

                    samples = sess.run(g_sample, feed_dict={z_pl: z_sample, y_pl: y_sample})
                    fig = plot(samples)
                    plt.savefig((result_path + "/Final_step_{}.png").format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

                n_class = np.zeros(10)

                n_class[0] = 5923
                n_class[1] = 6742
                n_class[2] = 5958
                n_class[3] = 6131
                n_class[4] = 5842
                n_class[5] = 5421
                n_class[6] = 5918
                n_class[7] = 6265
                n_class[8] = 5851
                n_class[9] = 5949

                n_image = int(sum(n_class))
                image_labels = np.zeros(shape=[n_image, len(n_class)])

                image_cntr = 0
                for class_cntr in np.arange(len(n_class)):
                    for cntr in np.arange(n_class[class_cntr]):
                        image_labels[image_cntr, class_cntr] = 1
                        image_cntr += 1

                z_sample = sample_z(n_image, Z_dim)

                images = sess.run(g_sample, feed_dict={z_pl: z_sample, y_pl: image_labels})

                print(f'saving genereated data of shape {images.shape} and {image_labels.shape}')
                save_str = 'dp-cgan-synth-code_balanced' if data_save_str is None else data_save_str
                np.savez(f'{save_str}-eps{max_target_eps}.npz', data=images, labels=image_labels)
                print('done saving')

                if run_auc_test:
                    assert dataset_key == 'digits'

                    x_test, y_test = loadlocal_mnist(
                        images_path='../../data/MNIST/raw/t10k-images-idx3-ubyte',
                        labels_path='../../data/MNIST/raw/t10k-labels-idx1-ubyte')

                    y_test = [int(y) for y in y_test]
                    result_file = open(result_path + "/" + "results.txt", "w")
                    print("Binarizing the labels ...")
                    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                    y_test = label_binarize(y_test, classes=classes)

                    print("\n################# Logistic Regression #######################")

                    print("  Classifying ...")
                    y_score = classify(images, image_labels, x_test, "lr", random_state_value=30)

                    print("  Computing ROC ...")
                    false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(y_test, y_score)
                    print("  AUROC: " + str(roc_auc["micro"]))
                    result_file.write("LR AUROC:  " + str(roc_auc["micro"]) + "\n")

                    print("\n################# Multi-layer Perceptron #######################")

                    print("  Classifying ...")
                    y_score = classify(images, image_labels, x_test, "mlp", random_state_value=30)

                    print("  Computing ROC ...")
                    false_positive_rate, true_positive_rate, roc_auc = compute_fpr_tpr_roc(y_test, y_score)
                    print("  AUROC: " + str(roc_auc["micro"]))
                    result_file.write("MLP AUROC:  " + str(roc_auc["micro"]) + "\n")

                step = num_training_steps
                break  # out of while loop, ending the function

            step = step + 1


def main():
    sigma_clipping_list = [[1.12, 1.1]]
    # sigma_clipping_list = [[0.1, 1.1]]
    batchSizeList = [600]
    epsilon = 9.6
    # epsilon = 1e10
    delta = 1e-5

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-save-str', type=str, default=None)  # set for custom save subdir
    parser.add_argument('--data', type=str, default='digits')  # options are digits and fashion
    ar = parser.parse_args()

    for iteration in range(1, 2):
        for sigma, clipping in sigma_clipping_list:
            for batchSize in batchSizeList:
                print("Running TensorFlow with Sigma=%f, Clipping=%d, batchSize=%d\n" % (sigma, clipping, batchSize))
                runTensorFlow(sigma, float(clipping), batchSize, epsilon, delta, iteration, ar.data_save_str,
                              ar.data, run_auc_test=False)


if __name__ == '__main__':
    main()


