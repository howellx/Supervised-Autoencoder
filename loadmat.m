test=readtable('mnist_test.csv');
train=readtable('mnist_train.csv');
test = test{:,:};
train = train{:,:};

autoenc_train = trainAutoencoder(train, 256, 'MaxEpochs',200);
train_data = encode(autoenc_train, train);

autoenc_test = trainAutoencoder(test, 256, 'MaxEpochs',200);
test_data = encode(autoenc_test, test);


% figure
% img = X_test(1:end, 2000)';
% imshow(reshape(img, img_size)) 
% title("Original");

save('mnist256_big.mat','test_data','train_data')