# Mini project plan

## Understanding of image classification task

The overview of a image classfication task consists of two steps. First, feed your model with massive training data including the matrix expressions of images along with their labels to modify the parameters of your model. Second, input your test image data and take the most possible label as the result.

What I should do is to decide the architecture of the whole model and the function of each layer. Several methods are applied to optimize the performance of the training, for example, batch normalization, regularization, and dropout. The basic CNN is organized as the form like CONV-RELU-CONV-RELU-POOL-FC. The champions of ImageNet developed much more complex models such as AlexNet and GoogleNet, which proved useful. 