"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *

class MLP(object):
  """
  This class implements a Multi-layer Perceptron in NumPy.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward and backward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes, neg_slope):
    """
    Initializes MLP object. 
    
    Args:
      n_inputs: number of inputs.
      n_hidden: list of ints, specifies the number of units
                in each linear layer. If the list is empty, the MLP
                will not have any linear layers, and the model
                will simply perform a multinomial logistic regression.
      n_classes: number of classes of the classification problem.
                 This number is required in order to specify the
                 output dimensions of the MLP
      neg_slope: negative slope parameter for LeakyReLU
    
    TODO:
    Implement initialization of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.n_inputs = n_inputs
    self.n_hidden = n_hidden
    self.n_classes = n_classes
    self.neg_slope = neg_slope
    self.layers = []

    n_neurons = [self.n_inputs] + self.n_hidden

    # for each hidden layer, we have Linear Module + ReLu module
    for i in range(len(n_hidden)):
        linear_module = LinearModule(in_features=n_neurons[i], out_features=n_neurons[i+1])
        self.layers.append(linear_module)
        leakeyrelu_module = LeakyReLUModule(neg_slope=self.neg_slope)
        self.layers.append(leakeyrelu_module)

    # for the output layer, we need Relu module + SoftMax moudle
    linear_module = LinearModule(in_features=n_neurons[-1], out_features=n_classes)
    self.layers.append(linear_module)
    softmax_module = SoftMaxModule()
    self.layers.append(softmax_module)
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Performs forward pass of the input. Here an input tensor x is transformed through 
    several layer transformations.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    
    TODO:
    Implement forward pass of the network.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for layer in self.layers:
      x = layer.forward(x)
    out = x
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Performs backward pass given the gradients of the loss. 

    Args:
      dout: gradients of the loss
    
    TODO:
    Implement backward pass of the network.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    for layer in reversed(self.layers):
      dout = layer.backward(dout)
    ########################
    # END OF YOUR CODE    #
    #######################

    return

if __name__ == '__main__':
    n_inputs = 6
    n_hidden = [3,4,5]
    n_classes = 3
    neg_slope = 0.1
    mlp = MLP(n_inputs, n_hidden, n_classes, neg_slope)
    x = np.random.randn(7, 6)
    mlp.forward(x)
    mlp.backward(np.random.normal(size=(7,3)))
