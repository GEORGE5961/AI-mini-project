import tensorflow as tf
import pickle
import numpy as np
import time

start = time.clock()

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='iso-8859-1')
    return dict

dict = unpickle('/Users/wuzhengyu/Desktop/github/AI-mini-project/AlexNet/cifar10/data_batch_1')
dict1 = unpickle('/Users/wuzhengyu/Desktop/github/AI-mini-project/AlexNet/cifar10/test_batch')

x_tr =  dict['data']
y_tr =  dict['labels']
x_tr = np.asarray(x_tr,dtype=np.float32) / 255.0
y_tr = np.asarray(y_tr,dtype=np.int32)

x_ev = dict1['data'] 
y_ev = dict1['labels']
x_ev = np.asarray(x_ev, dtype=np.float32) / 255.0
y_ev = np.asarray(y_ev, dtype=np.int32)
#xtr_rows = x_tr.reshape(x_tr.shape[0], 32 * 32 * 3) # xtr_rows becomes 50000 x 3072
#xte_rows = x_te.reshape(x_te.shape[0], 32 * 32 * 3) 

'''
def getOneHotLabel(label, depth):          #Usage: getOneHotLabel(label,depth=10)
    m = np.zeros([len(label), depth])
    for i in range(len(label)):
        m[i][label[i]] = 1
    return m
'''


class DataLoader():
    def __init__(self):
        self.train_data = x_tr                                 # np.array [10000, 3024]
        self.train_labels = y_tr                               # np.array [10000] of int32
        self.eval_data = x_ev                                  # np.array [10000, 784]
        self.eval_labels = y_ev                                # np.array [10000] of int32

    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_labels[index]

learning_rate = 1e-4                   # learning rate
num_epoch = 100                        # the num of epochs
batch_size = 1024                      # the num of images processed one time
dropout_rate = 0.5                     # the possibility of dropout
class_num = 10                         # the num of classes
display_step = 20                      # the steps of display
num_batches = 5                        

class AlexNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=24,             # 卷积核数目
            kernel_size=[3, 3],     # 感受野大小
            padding="same",         # padding策略
            activation=tf.nn.relu,   # 激活函数
            data_format="channels_last"
        )
        self.lrn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=96,             
            kernel_size=[3, 3],    
            padding="same",         
            activation=tf.nn.relu,
            data_format="channels_last"   
        )
        self.lrn2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)
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
            activation=tf.nn.relu,   
        )
        self.pool5 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.fc6 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc7 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.fc8 = tf.keras.layers.Dense(units=10)
        self.flatten = tf.keras.layers.Reshape(target_shape=(4 * 4 * 96,))

    def call(self,input):
        input = tf.reshape(input, [-1, 32, 32, 3]) 
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
        return tf.argmax(logits, axis=-1)     

model = AlexNet()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

'''
for batch_index in range(num_batches):
    X, y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pred)
        #print("Batch %d: loss %f" % (batch_index, loss))       
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
'''

X_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, 32*32*3), name='input')
y_placeholder = tf.placeholder(dtype=tf.int32, shape=(batch_size), name='label')
y_pred = model(X_placeholder)
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_placeholder, logits=y_pred)
train_op = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)  
    for batch_index in range(2):
        X, y = data_loader.get_batch(batch_size)
        sess.run(train_op, feed_dict={X_placeholder: X, y_placeholder: y})
        result = sess.run(merged,feed_dict={X_placeholder: X, y_placeholder: y})
        writer.add_summary(result,batch_index)
    
    num_eval_samples = np.shape(data_loader.eval_labels)[0]
    y_pred = model.predict(data_loader.eval_data)
    y_pred_np = y_pred.eval()
    print("Test accuracy: %f" % (np.sum(y_pred_np == data_loader.eval_labels) / num_eval_samples))

elapsed = (time.clock() - start)
print("Time used: %f seconds" % elapsed)