This supervised autoencoder repository attempts to train the MATLAB Deep Learning Toolbox autoencoder to reduce feature vectors based on classification as well as reconstruction error, by appending the label vector to the training data. 
There are two .m files that contain our code: 

•	encoder_custom_small256.m 
  o	This code will run with the smaller mnist256.mat dataset. 
  o	To run, simply run the file in the same directory as the mnist256.mat dataset.
  o	The parameters can be customized are at the beginning of the file. 
    •	noi_0 - digits you want to include in class -1, in a horizontal vector.
    •	noi_1 - digits you want to include in class +1, in a horizontal vector.
    •	img_size - dimensions of your image
    •	l - length of appended label vector
    •	a - scalar multiple of appended label vector
    •	epochs - number of epochs for the autoencoder
    •	display_num - test sample number you want to see the reconstruction of (somewhat arbituary)
    •	step_size - step size of k, the reduced dimension, between loop interations
o	The bottom as extra plots that can be uncommented if you want to more closely observe behavior of the LDA and SVM linear classifiers 

•	encoder_custom_big784
  o	This code will run with the larger mnist784.mat dataset. 
  o	To run, simply run the file in the same directory as the mnist_big.mat dataset.
  o	Otherwise, the code is the exact same as encoder_custom_small256.m 
