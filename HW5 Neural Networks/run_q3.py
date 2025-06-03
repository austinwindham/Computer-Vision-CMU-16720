import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 0.0025
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1], hidden_size, params, 'layer1')
Weights_initial = params['Wlayer1']
initialize_weights(hidden_size, train_y.shape[1], params, 'output')

training_loss = []
training_acc = []

valid_loss = []
valid_acc = []

#3.3 intial weights
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8, 8), axes_pad=0.0)
for i in range(64):
    grid[i].imshow(Weights_initial[:, i].reshape((32, 32)))

plt.show()

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h = forward(xb,params, 'layer1')
        probs = forward(h, params, 'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)

        total_loss += loss
        total_acc += acc

        # backward
        delta = probs-yb
        gradx = backwards(delta, params, 'output', linear_deriv)
        
        backwards(gradx, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1']-= learning_rate*params['grad_Wlayer1']
        params['blayer1']-= learning_rate*params['grad_blayer1']
        params['Woutput']-= learning_rate*params['grad_Woutput']
        params['boutput']-= learning_rate*params['grad_boutput']

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss/train_y.shape[0],total_acc/batch_num))

# run on validation set and report accuracy! should be above 75%
#valid_acc = None
##########################
##### your code here #####
##########################
    hvalid = forward(valid_x, params, 'layer1')
    probs_valid = forward(hvalid, params, 'output',softmax)
    loss_valid, acc_valid = compute_loss_and_acc(valid_y, probs_valid)
    # change loss based on piazza
    training_loss.append(total_loss/train_y.shape[0])
    valid_loss.append(loss_valid/valid_x.shape[0])
    # accuracies
    training_acc.append(total_acc/batch_num)
    valid_acc.append(acc_valid)

    # in loop to see if it is working

    print('Validation accuracy: ',acc_valid)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()


plt.figure(0)
plt.plot(np.arange(max_iters), training_loss,'r')
plt.plot(np.arange(max_iters), valid_loss, 'b')
plt.xlabel('Epoch')
plt.ylabel("Loss")
plt.legend(['Training', 'Valid'])

plt.figure(1)
plt.plot(np.arange(max_iters), training_acc,'r')
plt.plot(np.arange(max_iters), valid_acc, 'b')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend(['Training', 'Valid'])
plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# visualize weights here
##########################
##### your code here #####
##########################
#initial was put at the front, easiest method


Weight_layer1 = params['Wlayer1']
rows, cols = Weight_layer1.shape
fig3 = plt.figure()
grid = ImageGrid(fig3, 111, nrows_ncols=(8, 8), axes_pad=0.0)
for i in range(cols):
    grid[i].imshow(Weight_layer1[:, i].reshape((32, 32)))

plt.show()
print("past plot")
# Q3.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################
# get distribution, find max, see if it matches
for i in range(train_y.shape[0]):

    x = train_x[i,:].reshape((1, train_x.shape[1]))
    y = train_y[i,:].reshape((1, train_y.shape[1]))

    h = forward(x, params, 'layer1')
    probdis = forward(h, params,'output',softmax)
    #print(probdis.shape)
    x_loc = np.argmax(probdis[0,:])
    y_loc = np.where(y==1)[1][0]
    confusion_matrix[x_loc, y_loc] +=1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()