import pickle
import numpy as np
import matplotlib.pyplot as plt

import skimage
import skimage.io
import skimage.transform

import tensorflow as tf
import tensornets as nets 
import time

start = time.clock()  

def load_cifar10_batch(folder_path, batch_id):
    with open(folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='iso-8859-1')
        
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']    
    return features, labels

def getOneHotLabel(x):          
    encoded = np.zeros((len(x), 10),dtype=np.int8)
    
    for idx, val in enumerate(x):
        encoded[idx][val] = 1    
    return encoded


learning_rate = 1e-4                   # learning rate
num_epoch = 9                        # the num of epochs
batch_size = 8                      # the num of images processed one time
dropout_rate = 0.5                     # the possibility of dropout
class_num = 10                         # the num of classes
display_step = 20                      # the steps of display
num_batches = 5                        

def save_data(folder_path):  

    valid_features = []
    valid_labels = []
    for batch_index in range(0, num_batches):

        features, labels = load_cifar10_batch(folder_path, batch_index+1)
        labels = getOneHotLabel(labels)
        
        index_valid = int(len(features) * 0.1) 

        #   90% of the dataset: training data
        # - one_hot_encode the lables
        # - save in a new file named, "preprocess_batch_" + batch_number
        # - each file for each batch
        pickle.dump((features[:-index_valid], labels[:-index_valid]), open('preprocess_batch_' + str(batch_index+1) + '.p', 'wb'))

        #  unlike the training dataset, validation dataset will be added in a batch dataset
        #  10% of the dataset: validation data
        valid_features.extend(features[-index_valid:])
        valid_labels.extend(labels[-index_valid:])

    # save validation data
    valid_labels = getOneHotLabel(valid_labels)
    pickle.dump((np.array(valid_features), np.array(valid_labels)), open('preprocess_validation.p', 'wb'))

    # save test data
    with open(folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')

    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']

    test_labels = getOneHotLabel(test_labels)
    pickle.dump((np.array(test_features), np.array(test_labels)), open('preprocess_test.p', 'wb'))


def batch_features_labels(features, labels, batch_size):
    """
    Split features and labels into batches
    """
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def load_training_batch(batch_id, batch_size):
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    filename = 'preprocess_batch_' + str(batch_id) + '.p'
    features, labels = pickle.load(open(filename, mode='rb'))
    
    tmpFeatures = []
    
    for feature in features:
        tmpFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
        tmpFeatures.append(tmpFeature)

    return batch_features_labels(tmpFeatures, labels, batch_size)

#save_data('cifar10')

valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))

# resize: (None,32,32,3) --> (None,224,224,3)
new_valid_features = []
for feature in valid_features:
    tmpValidFeature = skimage.transform.resize(feature, (224, 224), mode='constant')
    new_valid_features.append(tmpValidFeature)
    
new_valid_features = np.array(new_valid_features)

x_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, 224,224,3), name='input')
y_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,10), name='label')

logits = nets.VGG19(x_placeholder, is_training=True, classes=10)
model = tf.identity(logits, name='logits')

loss = tf.losses.softmax_cross_entropy(y_placeholder, logits)
tf.summary.scalar("loss", loss)
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

correct_pred = tf.equal(tf.argmax(model, 1), tf.argmax(y_placeholder, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar("accuracy", accuracy)

logits.print_outputs()
print("-----------------------")
logits.print_summary()

merged = tf.summary.merge_all()

save_model_path = './image_classification'

print('Training...')
with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(logits.pretrained()) 
    print('model.pretrained ... done ... ')   
    for epoch in range(num_epoch): 
        for batch_index in range(1):
            for batch_features, batch_labels in load_training_batch(batch_index+1, batch_size):
                _, result = sess.run([train_op,merged], feed_dict={x_placeholder: batch_features, y_placeholder: batch_labels})
                writer.add_summary(result,batch_index+1)
            print('Epoch {:>2}, Batch {}:  '.format(epoch + 1, batch_index), end='')
    
            valid_acc = 0
            for batch_valid_features, batch_valid_labels in batch_features_labels(new_valid_features, valid_labels, batch_size):
                valid_acc += sess.run(accuracy, {x_placeholder:batch_valid_features, y_placeholder:batch_valid_labels})
            
            tmp_num = new_valid_features.shape[0]/batch_size
            print('Validation Accuracy: {:.5f}'.format(valid_acc/tmp_num))

print("-----------------------")
elapsed = (time.clock() - start)
print("Time used: %f seconds" % elapsed)
