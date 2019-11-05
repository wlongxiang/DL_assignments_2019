"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
  """
  Linear module. Applies a linear transformation to the input data. 
  """
  def __init__(self, in_features, out_features):
    """
    Initializes the parameters of the module. 
    
    Args:
      in_features: size of each input sample
      out_features: size of each output sample

    TODO:
    Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
    std = 0.0001. Initialize biases self.params['bias'] with 0. 
    
    Also, initialize gradients with zeros.
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.params = {'weight': None, 'bias': None}
    self.grads = {'weight': None, 'bias': None}
    # bias is of shape out_features x 1
    self.params["bias"] = np.zeros(out_features)
    # weight is of shape out_features x in_features
    self.params["weight"] = np.random.normal(loc=0, scale=0.0001, size=(out_features, in_features))
    # bias is of shape out_features x 1
    self.grads["bias"] = np.zeros(out_features)
    # weight is of shape out_features x in_features
    self.grads["weight"] = np.zeros((out_features, in_features))

    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """
    
    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    out = x @ self.params["weight"].T + self.params["bias"]
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module. Store gradient of the loss with respect to 
    layer parameters in self.grads['weight'] and self.grads['bias']. 
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # as we derivated from the practice
    self.grads['weight'] = dout.T @ self.x
    self.grads['bias'] = np.mean(dout, axis=0)
    dx = dout @ self.params['weight']
    ########################
    # END OF YOUR CODE    #
    #######################
    
    return dx

class LeakyReLUModule(object):
  """
  Leaky ReLU activation module.
  """
  def __init__(self, neg_slope):
    """
    Initializes the parameters of the module.

    Args:
      neg_slope: negative slope parameter.

    TODO:
    Initialize the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.a = neg_slope
    ########################
    # END OF YOUR CODE    #
    #######################

  def forward(self, x):
    """
    Forward pass.
    
    Args:
      x: input to the module
    Returns:
      out: output of the module
    
    TODO:
    Implement forward pass of the module. 
    
    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.                                                           #
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    self.x = x
    # this is the leaky relu implementation
    out = np.where(x > 0, x, self.a * x)
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.

    Args:
      dout: gradients of the previous module
    Returns:
      dx: gradients with respect to the input of the module
    
    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = dout * np.where(self.x< 0, self.a, 1)
    ########################
    # END OF YOUR CODE    #
    #######################    

    return dx


class SoftMaxModule(object):
  """
  Softmax activation module.
  """

  def forward(self, x):
    """
    Forward pass.
    Args:
      x: input to the module
    Returns:
      out: output of the module

    TODO:
    Implement forward pass of the module.
    To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    b= np.max(x, axis=1, keepdims=True)
    self.x = x
    out = np.exp(x-b) / np.sum(np.exp(x-b), axis=1, keepdims=True)
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, dout):
    """
    Backward pass.
    Args:
      dout: gradients of the previous modul
    Returns:
      dx: gradients with respect to the input of the module

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    dx = np.empty_like(dout)

    # for row in np.arange(len(self.out)):
    #   out_row_slice = self.out[np.newaxis, row, :]
    #   dout_row_slice = dout[np.newaxis, row, :]
    #   dx_row_slice = dout_row_slice @ (np.diag(out_row_slice) - out_row_slice.T @ out_row_slice)
    #   dx[row] = dx_row_slice

    # split into a list of rows
    tmp = np.split(self.out, dout.shape[0], axis=0)
    # make a list of diagonal matrix
    tmp = tmp * np.eye(self.out.shape[1])
    # calculate the full matrix from out
    tmp = tmp - self.out[:, :, np.newaxis] * self.out[:, np.newaxis, :]
    dx = np.squeeze(dout[:, np.newaxis, :] @ tmp)

    #######################
    # END OF YOUR CODE    #
    #######################

    return dx

class CrossEntropyModule(object):
  """
  Cross entropy loss module.
  """

  def forward(self, x, y):
    """
    Forward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      out: cross entropy loss

    Implement forward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # note that the small number is to avoid numerical issue when x is very small
    epsilon = 1e-9
    out = np.mean(-y * np.log(x + epsilon))
    self.out = out
    ########################
    # END OF YOUR CODE    #
    #######################

    return out

  def backward(self, x, y):
    """
    Backward pass.
    Args:
      x: input to the module
      y: labels of the input
    Returns:
      dx: gradient of the loss with the respect to the input x.

    TODO:
    Implement backward pass of the module.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    # note that the small number is to avoid numerical issue when x is very small
    epsilon = 1e-9
    dx = -y / (x+epsilon)
    # important to average the loss gradient by number of samples!
    dx = dx / len(x)
    ########################
    # END OF YOUR CODE    #
    #######################

    return dx