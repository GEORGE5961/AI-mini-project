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
display_step = 20                      # the steps of display

class AlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24,             # 卷积核数目
            kernel_size=[3, 3],     # 感受野大小
            padding="same",         # padding策略
            activation=tf.nn.relu   # 激活函数
        )
        self.lrn1 = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=96,             
            kernel_size=[3, 3],    
            padding="same",         
            activation=tf.nn.relu   
        )
        self.lrn2 = tf.keras.layers.BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=192,             
            kernel_size=[3, 3],     
            padding="same",         
            activation=tf.nn.relu   
        )
        self.conv4 = tf.keras.layers.Conv2D(
            filters=192,             
            kernel_size=[3, 3],     
            padding="same",         
            activation=tf.nn.relu   
        )
        self.conv5 = tf.keras.layers.Conv2D(
            filters=96,             
            kernel_size=[3, 3],     
            padding="same",         
            activation=tf.nn.relu   
        )
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.fc6 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc7 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc8 = tf.keras.layers.Dense(units=10)
        self.flatten = tf.keras.layers.Reshape(target_shape=(4 * 4 * 96,))

    def call(self,input):
        input = tf.reshape(input, [-1, 32, 32, 1])  
        x = self.conv1(input)   #[batch_size, 32, 32, 24]
        x = self.lrn1(x)        #[batch_size, 32, 32, 24]
        x = self.pool1(x)       #[batch_size, 16, 16, 24]
        x = self.conv2(x)       #[batch_size, 16, 16, 96]
        x = self.lrn2(x)        #[batch_size, 16, 16, 96]
        x = self.pool2(x)       #[batch_size, 8, 8, 96]
        x = self.conv3(x)       #[batch_size, 8, 8, 192]
        x = self.conv4(x)       #[batch_size, 8, 8, 192]
        x = self.conv5(x)       #[batch_size, 8, 8, 96]
        x = self.pool5(x)       #[batch_size, 4, 4, 96]
        x = self.flatten(x)     #[batch_size, 4*4*96]
        x = self.fc6(x)         # [batch_size, 1024]
        x = self.fc7(x)         # [batch_size, 1024]
        x = self.fc8(x)         # [batch_size, 10]
        return x

    def predict(self, inputs):
        logits = self(inputs)




        
