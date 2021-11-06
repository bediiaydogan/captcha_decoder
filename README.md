Deep Learning Based CAPTCHA Solver

CAPTCHA stands for Completely Automated Public Turing test to tell Computers and Humans Apart. CAPTCHA determines whether the user is real or a spam robot. Automatic decoding of CAPTCHA is a critical issue for web scraping. The purpose of this deep learning based CAPTCHA solver is to automate scraping of a particular website. 


Dataset Description

The dataset consists of 1862 PNG images of text based CAPTCHA. 1490 images have been used for training and the remaining images for testing purpose. Each image is of 5 character set and the character set is defined as small letters and digits. The dimensions of the CAPTCHA images are 50x200. The images consist of noise in the form of bold lines crossing characters.


Preprocessing

Noise removal and character segmentation are the two main steps in preprocessing of CAPTCHA images. 

First, the RGB image is converted to a greyscale image; all pixel values are in the range 0–255. Next, the background color is converted to white by detecting character boundaries (by detecting changes in RGB values for each pixel in a column). 

Once the image is converted to greyscale and background is converted to white, the next step of preprocessing is removing noise. As mentioned earlier, a bold line is present in the image that is crossing characters. Lines are removed from images by converting RGB values < 10 are converted to 255 (white). Next, the images are converted to black-white by converting RGB values < 255 to 0 (black).

The last step in preprocessing is character segmentation. The individual characters are segmented out of the image in order to train a model on the character classification task. The characters are mostly non-intersecting. But, there are some few cases that characters are touching each other. The segmentation step is the most challenging step in preprocessing and tried to be clarified in code comments. 

Preprocessing steps are illustrated below. CAPTCHA images do not have any borders. Borders around images are added for clarification.


Model

The model developed for the CAPTCHA solver uses Convolutional Neural Network. It consists of the input layer, convolutional layers, max pooling layers, flatten layers, dropout layers and dense layers. An Adam optimizer is used with cross-entropy loss. A validation split of 0.2 is used. The batch size used is 32. While the model was trained with 10 epochs, the model typically converged in 5 epochs. All character images passed into the network are size 40x50x1. The accuracy obtained after 10 epochs is 0.9887.


 1/56 [..............................] - ETA: 7s - loss: 1.8775e-06 - accuracy: 1.0000
 3/56 [>.............................] - ETA: 1s - loss: 6.6801e-04 - accuracy: 1.0000
 4/56 [=>............................] - ETA: 3s - loss: 8.9367e-04 - accuracy: 1.0000
 6/56 [==>...........................] - ETA: 2s - loss: 0.0033 - accuracy: 1.0000    
 8/56 [===>..........................] - ETA: 2s - loss: 0.0026 - accuracy: 1.0000
10/56 [====>.........................] - ETA: 2s - loss: 0.0055 - accuracy: 0.9969
12/56 [=====>........................] - ETA: 1s - loss: 0.0046 - accuracy: 0.9974
14/56 [======>.......................] - ETA: 1s - loss: 0.0039 - accuracy: 0.9978
16/56 [=======>......................] - ETA: 1s - loss: 0.0045 - accuracy: 0.9980
18/56 [========>.....................] - ETA: 1s - loss: 0.0045 - accuracy: 0.9983
21/56 [==========>...................] - ETA: 1s - loss: 0.0216 - accuracy: 0.9955
23/56 [===========>..................] - ETA: 1s - loss: 0.0229 - accuracy: 0.9946
25/56 [============>.................] - ETA: 1s - loss: 0.0211 - accuracy: 0.9950
27/56 [=============>................] - ETA: 1s - loss: 0.0196 - accuracy: 0.9954
29/56 [==============>...............] - ETA: 1s - loss: 0.0437 - accuracy: 0.9935
31/56 [===============>..............] - ETA: 0s - loss: 0.0413 - accuracy: 0.9940
33/56 [================>.............] - ETA: 0s - loss: 0.0388 - accuracy: 0.9943
35/56 [=================>............] - ETA: 0s - loss: 0.0432 - accuracy: 0.9929
37/56 [==================>...........] - ETA: 0s - loss: 0.0448 - accuracy: 0.9924
39/56 [===================>..........] - ETA: 0s - loss: 0.0588 - accuracy: 0.9912
41/56 [====================>.........] - ETA: 0s - loss: 0.0747 - accuracy: 0.9901
43/56 [======================>.......] - ETA: 0s - loss: 0.0713 - accuracy: 0.9906
45/56 [=======================>......] - ETA: 0s - loss: 0.0762 - accuracy: 0.9903
47/56 [========================>.....] - ETA: 0s - loss: 0.0847 - accuracy: 0.9894
49/56 [=========================>....] - ETA: 0s - loss: 0.0889 - accuracy: 0.9879
51/56 [==========================>...] - ETA: 0s - loss: 0.0855 - accuracy: 0.9884
53/56 [===========================>..] - ETA: 0s - loss: 0.0846 - accuracy: 0.9882
55/56 [============================>.] - ETA: 0s - loss: 0.0815 - accuracy: 0.9886
56/56 [==============================] - 2s 35ms/step - loss: 0.0811 - accuracy: 0.9887


Labels:

{'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 'a': 7, 'b': 8, 'c': 9, 'd': 10, 'e': 11, 'f': 12, 'g': 13, 'h': 14, 'k': 15, 'm': 16, 'n': 17, 'p': 18, 'r': 19, 'w': 20, 'x': 21, 'y': 22}

Confusion Matrix:

tf.Tensor(
[[74  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0 80  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0 83  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  1  0 77  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0 78  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0 77  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0 75  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0  0 76  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0 78  0  0  0  0  0  0  0  0  1  1  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0 82  0  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0 76  0  0  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0 67  0  0  0  0  1  0  0  1  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0 64  0  0  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0 82  0  0  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0  0  0  1  0  0  0  0  0 73  0  0  0  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  1  0  0  0  1  0  0 78  0  0  0  0  0  0  0]
 [ 1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 70  1  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 83  0  0  0  0  0]
 [ 0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0 76  0  0  0  0]
 [ 2  0  0  0  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0 71  0  0  0]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 79  0  1]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 74  0]
 [ 0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 78]], shape=(23, 23), dtype=int32)
