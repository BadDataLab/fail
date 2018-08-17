import numpy as np
import os
import sys

from testbed import testbed
from examples.jsma_experiments import JSMAUnconstrainedExperiment
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_eval, model_argmax, model_train
from dataset import Dataset
import tensorflow as tf

from cleverhans.loss import LossCrossEntropy
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import train, model_eval, model_argmax
from cleverhans_tutorials.tutorial_models import ModelBasicCNN


def one_hot_encoder(ys, nb_classes):
    ohe_ys = []
    for y in ys:
        ohe_y = np.zeros(nb_classes)
        ohe_y[y] = 1

        ohe_ys.append(ohe_y)

    return np.array(ohe_ys)

class CNNModel:

    def __init__(self,scope):
        # Define input TF placeholder
        self.scope = scope
        self.reset()

    def reset(self):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            self.sess = tf.get_default_session()
            # Define input TF placeholder
            self.x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
            self.y = tf.placeholder(tf.float32, shape=(None, 10))
            self.model = ModelBasicCNN('model1', 10, 64)
            self.preds = self.model.get_logits(self.x)
            self.loss = LossCrossEntropy(self.model, smoothing=0.1)

    def train(self, dataset):
        train_params = {
            'nb_epochs': 1,
            'batch_size': 32,
            'learning_rate': 1e-2
        }
        with tf.variable_scope(self.scope,reuse=tf.AUTO_REUSE):
            self.sess.run(tf.global_variables_initializer())
            train(self.sess, self.loss, self.x, self.y, dataset.x, dataset.y, args=train_params)

    def test(self, dataset):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            eval_params = {'batch_size': 32}
            accuracy = model_eval(self.sess, self.x, self.y, self.preds, dataset.x, dataset.y, args=eval_params)
            print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
            return {'accuracy': accuracy}

    def tf(self):
        return self.model


if __name__ == '__main__':
        #with testbed() as tb:
        tb = testbed()
        x_train, y_train, x_test, y_test = data_mnist()
        sess = tf.InteractiveSession()

        victim = CNNModel('victim')
        surrogate = CNNModel('surrogate')

        mnist = Dataset(np.append(x_train, x_test, axis=0), np.append(y_train, y_test, axis=0), train_pct=0.1)
        tb.register_dataset(mnist)
        tb.register_victim(victim)
        tb.register_surrogate(surrogate)

        e1 = JSMAUnconstrainedExperiment(['some_label'])
        # e2 = JSMALimitedFeatureKnowledgeExperiment(border=4)
        tb.register_experiment(e1)

        tb.run_experiments(runs=2, out_dir=os.path.join('output', 'jsma-FAIL'))
