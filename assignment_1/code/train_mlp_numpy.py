"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import time

import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule, LinearModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None

def accuracy(predictions, targets):
  """
  Computes the prediction accuracy, i.e. the average of correct predictions
  of the network.
  
  Args:
    predictions: 2D float array of size [batch_size, n_classes]
    labels: 2D int array of size [batch_size, n_classes]
            with one-hot encoding. Ground truth labels for
            each sample in the batch
  Returns:
    accuracy: scalar float, the accuracy of predictions,
              i.e. the average correct predictions over the whole batch
  
  TODO:
  Implement accuracy computation.
  """

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  number_of_inputs = len(targets)
  y_pred = np.argmax(predictions, axis=1)
  y_true = np.argmax(targets, axis=1)
  accuracy = np.sum(y_pred == y_true)/number_of_inputs
  ########################
  # END OF YOUR CODE    #
  #######################

  return accuracy

def train():
  """
  Performs training and evaluation of MLP model. 

  TODO:
  Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
  """

  ### DO NOT CHANGE SEEDS!
  # Set the random seeds for reproducibility
  np.random.seed(42)

  ## Prepare all functions
  # Get number of units in each hidden layer specified in the string such as 100,100
  if FLAGS.dnn_hidden_units:
    dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
    dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
  else:
    dnn_hidden_units = []

  # Get negative slope parameter for LeakyReLU
  neg_slope = FLAGS.neg_slope

  ########################
  # PUT YOUR CODE HERE  #
  #######################
  lr = FLAGS.learning_rate
  eval_freq= FLAGS.eval_freq
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  input_size = 32*32*3
  output_size = 10
  # load dataset
  raw_data = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
  train_data = raw_data['train']
  validation_data = raw_data["validation"]
  test_data = raw_data['test']

  model = MLP(n_inputs=input_size, n_hidden=dnn_hidden_units, n_classes=output_size, neg_slope=neg_slope)
  loss_target = CrossEntropyModule()
  csv_data = [['step', 'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy']]
  for step in range(max_steps):
    x, y = train_data.next_batch(batch_size)
    x = x.reshape(batch_size, input_size)
    # train
    output = model.forward(x)
    loss_avg = loss_target.forward(output, y)
    dout = loss_target.backward(output, y)
    model.backward(dout)
    # only need to update weights for linear module for each step
    for layer in model.layers:
      if isinstance(layer, LinearModule):
        layer.params['weight'] -= lr * layer.grads['weight']
        layer.params['bias'] -= lr * layer.grads['bias']
    train_acc = accuracy(output, y)
    # with the \r and end = '' trick, we can print on the same line
    print('\r[{}/{}] train_loss: {}  train_accuracy: {}'.format(step + 1, max_steps,round(loss_avg, 3),round(train_acc, 3)), end='')
    # evaluate
    if step % eval_freq == 0 or step >= (max_steps - 1):
      x, y = test_data.next_batch(test_data.num_examples)
      x = x.reshape(test_data.num_examples, input_size)
      output = model.forward(x)
      test_loss = loss_target.forward(output, y)
      test_acc = accuracy(output, y)
      csv_data.append([step, loss_avg, test_loss, train_acc, test_acc])
      print(' test_loss: {}, test_accuracy: {}'.format(round(test_loss, 3), round(test_acc,3)))
  with open('results/train_summary_np_{}.csv'.format(int(time.time())), 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)
  ########################
  # END OF YOUR CODE    #
  #######################

def print_flags():
  """
  Prints all entries in FLAGS variable.
  """
  for key, value in vars(FLAGS).items():
    print(key + ' : ' + str(value))

def main():
  """
  Main function
  """
  # Print all Flags to confirm parameter settings
  print_flags()

  if not os.path.exists(FLAGS.data_dir):
    os.makedirs(FLAGS.data_dir)

  # Run the training operation
  train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  parser.add_argument('--neg_slope', type=float, default=NEG_SLOPE_DEFAULT,
                      help='Negative slope parameter for LeakyReLU')
  FLAGS, unparsed = parser.parse_known_args()

  main()