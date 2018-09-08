import tensorflow as tf
import pickle
import numpy as np

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='iso-8859-1')
    return dict

dict = unpickle('/Users/wuzhengyu/Desktop/github/AI-mini-project/AlexNet/cifar10/data_batch_1')
dict1 = unpickle('/Users/wuzhengyu/Desktop/github/AI-mini-project/AlexNet/cifar10/test_batch')

x_tr =  dict['data']
y_tr =  dict['labels']
x_tr = np.array(x_tr)
y_tr = np.array(y_tr)

x_te = dict1['data']
y_te = dict1['labels']
print(x_te.shape)
xtr_rows = x_tr.reshape(x_tr.shape[0], 32 * 32 * 3) # xtr_rows becomes 50000 x 3072
xte_rows = x_te.reshape(x_te.shape[0], 32 * 32 * 3) 

learning_rate = 1e-4                   # learning rate
num_epoch = 100                        # the num of epochs
batch_size = 1024                      # the num of images processed one time
dropout_rate = 0.5                     # the possibility of dropout
class_num = 10                         # the num of classes
train_layers = ['fc8', 'fc7', 'fc6']   # train layers
display_step = 20                      # the steps of display

class AlexNetModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,             # 卷积核数目
            kernel_size=[5, 5],     # 感受野大小
            padding="same",         # padding策略
            activation=tf.nn.relu   # 激活函数
        )
