# Neural-Network-Project
ECS659P/ECS7026P Neural Networks and Deep Learning - Project Completed for this module is presented here. 

# Neural Networks and Deep Learning (ECS659P/ECS7026P)
In this assignment, you will implement a neural network that classifies images. Please read this entire document before you start working on the assignment.

## 1. Dataset
The CIFAR-10 dataset is composed of 60000 small (3×32×32) color images, each of which belongs to one of 10 classes. There are 6000 images per class. The images are divided into a training dataset composed of 50000 examples and a testing dataset composed of 10000 examples. This dataset is readily available for PyTorch.

![CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
*Figure 1: Examples from CIFAR-10 (classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).*

Your first task is to create a DataLoader for the training dataset and a DataLoader for the testing dataset, which should enable generating batches of examples.

## 2. Basic architecture
Your next task is to implement the neural network architecture described in this section to classify images from the CIFAR-10 dataset. The architecture is composed of a sequence of intermediate blocks B1, B2, . . . , BK that are followed by an output block O, as shown in Figure 2. These blocks are detailed in the following subsections.

### 2.1 Intermediate block
An intermediate block receives an image x and outputs an image x′. Each block has L independent convolutional layers. Each convolutional layer Cl in a block receives the input image x and outputs an image Cl(x). Each of these images is combined into the single output image x′, which is given by

x′ = a1C1(x) + a2C2(x) + . . .+ aLCL(x) 

where a = [a1, a2, . . . , aL]^T is a vector that is also computed by the block. Note that each convolutional layer in a block receives the same input image x (and not the output of another convolutional layer within the block).

Suppose that the input image x has c channels. In order to compute the vector a, the average value of each channel of x is computed and stored into a c-dimensional vector m. The vector m is the input to a fully connected layer that outputs the vector a. Note that this fully connected layer should have as many units as there are convolutional layers in the block.

Each block in the basic architecture may have a different number of convolutional layers, and each convolutional layer may have different hyperparameters (within or across blocks). However, every convolutional layer within a block should output an image with the same shape.

### 2.2 Output block
The output block receives an image x (output of the last intermediate block) and outputs a logits vector o. Suppose that the input image x has c channels. In order to compute the vector o, the average value of each channel of x is computed and stored into a c-dimensional vector m. The vector m is the input to a sequence of zero or more fully connected layer(s) that output the vector o.

## 3. Training and testing
Your next task is to train a neural network with the basic architecture described in the previous section and compute its accuracy in the testing dataset.

For a given batch size b, your network should receive a b×3×32×32 tensor composed of b images of shape 3×32×32 and output a b×10 logits matrix O. You should use a cross entropy loss for training.

You should make the remaining decisions (such as hyperparameter settings) by yourself. You may adapt the code provided in previous lectures to compute evaluation metrics and implement the training procedure.

*Hint: Before implementing the basic architecture described in the previous section, you may want to implement a simpler architecture that allows testing whether the remaining components are working. Your experiments may also benefit from using a GPU through Google Colab (Runtime > Change runtime type).*

## 4. Improving the results
Your next task is to improve the results of your initial implementation. You should train different neural networks in the training dataset to find a neural network that achieves the highest possible accuracy in the testing dataset. Your mark will depend on this accuracy.

Note that you are being asked to use the testing dataset as if it were a validation dataset in order to simplify this assignment. This is generally a methodological mistake, since a testing dataset should only be used to assess generalization after the hyperparameters are chosen (rather than used to choose hyperparameters).

In order to improve the results of your initial implementation, you can try different hyperparameters and employ all the techniques covered during Week 6 and Week 8. You can also make mild changes to the basic architecture, such as including additional layers between intermediate blocks. In order to inspire your solution, you are strongly encouraged to find and study existing code that implements neural networks for (CIFAR-10) image classification.

*You are not allowed to radically change the basic architecture described in Section 2. For example, each intermediate block must weight the outputs of independent convolutional layers using coefficients obtained by a fully connected layer that receives the average value of each channel of the input image. As a general rule, reverting your implementation to an implementation of the basic architecture described in Section 2 should be trivial.*

If you are not sure whether a change would be considered too radical, please ask. This assignment is intended to assess whether you are able to implement an architecture based on a high-level description (rather than copying or adapting existing code). Therefore, you will receive no marks if you implement an architecture that is not clearly based on the basic architecture described in Section 2, regardless of its performance.

You should store the following statistics about the neural network that achieves the highest accuracy in the testing dataset: loss for each training batch, training accuracy after each epoch, and testing accuracy after each epoch. You should plot the loss for each training batch. You should also plot the training accuracy and testing accuracy for each training epoch.
