import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    
    #eqn 16
    bounds = np.sqrt(6/(in_size +out_size))
    
    W = np.random.uniform(-1*bounds, bounds,(in_size, out_size))
    b = np.zeros(out_size)

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    
    res = 1/(1 + np.exp(-x))

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]



    pre_act = X@W +b
    post_act = activation(pre_act)


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    
    
    s_i = np.exp(x -np.max(x, axis=1, keepdims=True))
    res = s_i / s_i.sum(axis=1, keepdims=True)

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    

    correct = 0
    for i in range(y.shape[0]):

        cl = np.argmax(probs[i,:])

        if y[i, cl] == 1:
            correct+= 1

    acc = correct/y.shape[0]
    
    loss = -1*np.sum(y*np.log(probs))

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    
    act_der= activation_deriv(post_act)

    grad_W = X.T @ (act_der*delta)
    # might need to fix
    grad_b = np.sum(act_der*delta, axis=0)
    grad_X = act_der*delta @ W.T

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []

    # Shuffle the data indices
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    # Create batches
    for i in range(0, x.shape[0], batch_size):
        batch_indices = indices[i:i+batch_size]
        bx = x[batch_indices, :]
        by = y[batch_indices, :]
        batches.append((bx, by))

    return batches
