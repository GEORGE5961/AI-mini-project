import pickle
from scipy.misc import imread, imsave,imresize
import matplotlib.pyplot as plt
import numpy as np
def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='iso-8859-1')
    return dict


'''
class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
'''
dict = unpickle('data_batch_1')
dict1 = unpickle('test_batch')

x_tr =  dict['data']
y_tr =  dict['labels']
x_tr = np.array(x_tr)
y_tr = np.array(y_tr)

x_te = dict1['data']
y_te = dict1['labels']

xtr_rows = x_tr.reshape(x_tr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072
xte_rows = x_te.reshape(x_te.shape[0], 32 * 32 * 3) 

#nn = NearestNeighbor() # create a Nearest Neighbor classifier class
#nn.train(xtr_rows, y_tr) # train the classifier on the training images and labels
#yte_predict = nn.predict(xte_rows) # predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
#print 'Accuracy: %f' % ( np.mean(yte_predict == y_te) )
#show a picture
'''
x_tr = np.reshape(x_tr,(10000,3,32,32))
red = x_tr[0][0].reshape(1024,1)
green = x_tr[0][1].reshape(1024,1)
blue = x_tr[0][2].reshape(1024,1)
img = np.concatenate((red,green,blue),axis=1)
img = img.reshape(32,32,3)
plt.imshow(img)
plt.show()
'''
