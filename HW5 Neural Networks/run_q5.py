import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
initialize_weights(hidden_size,hidden_size, params, 'hidden')
initialize_weights(hidden_size,hidden_size, params, 'hidden2')
initialize_weights(hidden_size, train_x.shape[1], params,'output')

training_loss =[]
# m+name convention in comment below
keylist = [i for i in params.keys()]

for n in keylist:
    params['m_'+n] = np.zeros(params[n].shape)

# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        # relu, relu, relu,sigmoid
        h = forward(xb, params, 'layer1',relu)
        h2 = forward(h, params, 'hidden',relu)
        h3 = forward(h2, params, 'hidden2',relu)
        y = forward(h3, params, 'output', sigmoid)

        # from comments 
        total_loss += np.sum((xb - y)**2)
        delta = -2*(xb-y)


        d1 = backwards(delta, params, 'output', sigmoid_deriv)
        d2 = backwards(d1, params, 'hidden2',relu_deriv)
        d3 = backwards(d2, params, 'hidden',relu_deriv)
        backwards(d3, params,'layer1',relu_deriv)

        # implement momentum
        for n in params.keys():
            
            if '_' in n:
                continue

            params['m_'+n] = 0.9*params['m_'+n]-learning_rate*params['grad_'+n]
            params[n] += params['m_'+n]

    training_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss/train_x.shape[0]))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
# Q5.3.1 And 5.2
import matplotlib.pyplot as plt

plt.figure(0)
plt.plot(np.arange(max_iters), np.array(training_loss)/train_x.shape[0],'r')
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.show()

# visualize some results
##########################
##### your code here #####
##########################
# same relu, relu, relu, sigmoid
h = forward(valid_x,params,'layer1',relu)
h1 = forward(h,params,'hidden',relu)
h2 = forward(h1,params,'hidden2',relu)
y = forward(h2,params,'output',sigmoid)

# 3600 a, b, 0, 6, 8
printouts = [19, 20, 110, 120, 2610, 2620, 3211, 3220, 3411, 3420]

for n in printouts:

    plt.subplot(1,2,1)
    plt.imshow(valid_x[n].reshape(32,32).T)
    plt.title("Validation Image")

    plt.subplot(1,2,2)
    plt.imshow(y[n].reshape(32,32).T)
    plt.title("Reconstructed Image")
    plt.show()



# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
##########################
##### your code here #####
##########################

psnr = [peak_signal_noise_ratio(valid_x[n], y[n]) for n in range(y.shape[0])]
print('psnr: ')
print(np.array(psnr).mean())
