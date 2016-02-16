import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  num_training = X.shape[0]
  num_classes = W.shape[1]
  num_dim = X.shape[1]
  scaled_scores = scores - np.reshape(np.max(scores, axis=1), (num_training, -1))
  correct_scores = scaled_scores[np.arange(scaled_scores.shape[0]), y]
  L = -correct_scores + np.log(np.sum(np.exp(scaled_scores), axis=1))
  loss = np.mean(L) + reg * np.sum(W**2)
  K= np.zeros((num_training, num_classes))
  K[np.arange(num_training), y] = -1
  scores = scaled_scores
  exp_scores = np.exp(scores)
  sum_exp_scores = np.sum(exp_scores, axis=1)
  probabilities = exp_scores/np.reshape(sum_exp_scores, (sum_exp_scores.shape[0], -1))
  added_with_K = K + probabilities
  dW += X.T.dot(added_with_K)
  dW /= num_training
  dW += 2 * reg * W

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return softmax_loss_naive(W, X, y, reg)

