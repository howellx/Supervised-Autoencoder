%% Parameters
noi_0 = [3];            %Digits in Class 1 (+1)
noi_1 = [8];            %Digits in Class 2 (-1)
img_size = [16, 16];        %Image dimensions
l = 20;                     %Length of label vector
a = 1;                      %Label vector scalar
epochs = 10;                %Autoencoder epochs
display_num = 200;          %Test point to display

%% Setting data up
% load mnist into workspace
mnist = load('mnist256.mat').mnist;
label = mnist(:,1);
X = mnist(:,(2:end));

%Setting up training and testing data and labels
num_idx0 = zeros(length(label), 1);
for i = noi_0
    num_idx0 = num_idx0 + (label == i);
end
num_idx0 = logical(num_idx0);

num_idx1 = zeros(length(label), 1);
for i = noi_1
    num_idx1 = num_idx1 + (label == i);
end
num_idx1 = logical(num_idx1);


X0 = X(num_idx0,:);
X1 = X(num_idx1,:);

n0_temp = floor(3*length(X0) / 4);
n1_temp = floor(3*length(X1) / 4);
X_train = [X0(1:n0_temp, :); X1(1:n1_temp, :)]';
X_test = [X0(n0_temp+1:end, :); X1(n1_temp+1:end, :)]';

[d,n_train] = size(X_train);
[~,n_test] = size(X_test);

temp0 = label(num_idx0);
temp1 = label(num_idx1);
Y_train = [zeros(n0_temp, 1) + 1; zeros(n1_temp, 1) - 1]; 
Y_train_ext = Y_train; %[Y_train; Y_train];
Y_test = [zeros(length(temp0) - n0_temp, 1) + 1; zeros(length(temp1) - n1_temp, 1) - 1];

X_train_custom = [X_train; repmat(Y_train', l, 1)]; %X_train_custom = [X_train, X_train; repmat(Y_train', l, 1), zeros(l, n_train)];
X_test_custom = [X_test; zeros(l, n_test)];
[d_custom,n_train_custom] = size(X_train_custom);
[~,n_test_custom] = size(X_test_custom);



%% Creating CCR and MSE vecs for plotting
%
CCR_train_lda_vec = [];
CCR_test_lda_vec = [];

CCR_train_svmLin_vec = [];
CCR_test_svmLin_vec = [];

CCR_train_svmKern_vec = [];
CCR_test_svmKern_vec = [];

MSE_train_vec = [];
MSE_test_vec = [];

CCR_train_lda_vec_reconstructed = [];
CCR_test_lda_vec_reconstructed = [];

CCR_train_svmLin_vec_reconstructed = [];
CCR_test_svmLin_vec_reconstructed = [];

CCR_train_svmKern_vec_reconstructed = [];
CCR_test_svmKern_vec_reconstructed = [];



CCR_train_lda_vec_custom = [];
CCR_test_lda_vec_custom = [];

CCR_train_svmLin_vec_custom = [];
CCR_test_svmLin_vec_custom = [];

CCR_train_svmKern_vec_custom = [];
CCR_test_svmKern_vec_custom = [];

MSE_train_vec_custom = [];
MSE_test_vec_custom = [];

CCR_train_lda_vec_custom_reconstructed = [];
CCR_test_lda_vec_custom_reconstructed = [];

CCR_train_svmLin_vec_custom_reconstructed = [];
CCR_test_svmLin_vec_custom_reconstructed = [];

CCR_train_svmKern_vec_custom_reconstructed = [];
CCR_test_svmKern_vec_custom_reconstructed = [];

CCR_test_bottomrows_custom_reconstructed = [];


CCR_train_svmKern_vec_custom_0s = [];
CCR_test_svmKern_vec_custom_0s = [];
MSE_train_vec_custom_0s = [];
MSE_test_vec_custom_0s = [];


%% Starting the loop
positionCount = 1;
k_array = 1:10:d;
for k = k_array
    
    %% Unmodified autoencoder with regular inputs
    %autoencoding
    autoenc = trainAutoencoder(X_train, k, 'MaxEpochs',epochs);
    
    X_train_small = encode(autoenc, X_train);
    X_test_small = encode(autoenc, X_test);
    
    k = k
    
    %lda
    lda = fitcdiscr(X_train_small', Y_train);
    Y_hat_train = predict(lda, X_train_small');
    Y_hat_test = predict(lda, X_test_small');

    CCR_train = sum(Y_hat_train == Y_train)/n_train;
    CCR_test = sum(Y_hat_test == Y_test)/n_test;

    CCR_train_lda_vec = [CCR_train_lda_vec, CCR_train];
    CCR_test_lda_vec = [CCR_test_lda_vec, CCR_test];


    %svm linear
    svm_lin = fitclinear(X_train_small',Y_train);
    Y_hat_train = predict(svm_lin, X_train_small');
    Y_hat_test = predict(svm_lin, X_test_small');

    CCR_train = sum(Y_hat_train == Y_train)/n_train;
    CCR_test = sum(Y_hat_test == Y_test)/n_test;

    CCR_train_svmLin_vec = [CCR_train_svmLin_vec, CCR_train];
    CCR_test_svmLin_vec = [CCR_test_svmLin_vec, CCR_test];

    % svm kernel
    svm_kernel = fitckernel(X_train_small',Y_train);
    Y_hat_train = predict(svm_kernel, X_train_small');
    Y_hat_test = predict(svm_kernel, X_test_small');

    CCR_train = sum(Y_hat_train == Y_train)/n_train;
    CCR_test = sum(Y_hat_test == Y_test)/n_test;

    CCR_train_svmKern_vec = [CCR_train_svmKern_vec, CCR_train];
    CCR_test_svmKern_vec = [CCR_test_svmKern_vec, CCR_test];
    
    % MSE
    X_train_Reconstructed = predict(autoenc,X_train);
    X_test_Reconstructed = predict(autoenc,X_test);
    
    mseError_train = mse(X_train - X_train_Reconstructed);
    MSE_train_vec = [MSE_train_vec, mseError_train];
    
    mseError_test = mse(X_test - X_test_Reconstructed);
    MSE_test_vec = [MSE_test_vec, mseError_test];



    %% Modified autoencoder with custom inputs
    %autoencoding
    autoenc_custom = trainAutoencoder(X_train_custom, k, 'MaxEpochs',epochs);
    
    X_train_small_custom = encode(autoenc_custom, X_train_custom);
    X_test_small_custom = encode(autoenc_custom, X_test_custom);
    
    
    %lda
    lda_custom = fitcdiscr(X_train_small_custom', Y_train_ext);
    Y_hat_train = predict(lda_custom, X_train_small_custom');
    Y_hat_test = predict(lda_custom, X_test_small_custom');

    CCR_train = sum(Y_hat_train == Y_train_ext)/n_train_custom;
    CCR_test = sum(Y_hat_test == Y_test)/n_test_custom;

    CCR_train_lda_vec_custom = [CCR_train_lda_vec_custom, CCR_train];
    CCR_test_lda_vec_custom = [CCR_test_lda_vec_custom, CCR_test];

    %svm linear
    svm_lin_custom = fitclinear(X_train_small_custom',Y_train_ext);
    Y_hat_train_custom = predict(svm_lin_custom, X_train_small_custom');
    Y_hat_test_custom = predict(svm_lin_custom, X_test_small_custom');

    CCR_train = sum(Y_hat_train_custom == Y_train_ext)/n_train;
    CCR_test = sum(Y_hat_test_custom == Y_test)/n_test;

    CCR_train_svmLin_vec_custom = [CCR_train_svmLin_vec_custom, CCR_train];
    CCR_test_svmLin_vec_custom = [CCR_test_svmLin_vec_custom, CCR_test];

    %svm kernel
    svm_kernel_custom = fitckernel(X_train_small_custom',Y_train);
    Y_hat_train_custom = predict(svm_kernel_custom, X_train_small_custom');
    Y_hat_test_custom = predict(svm_kernel_custom, X_test_small_custom');

    CCR_train = sum(Y_hat_train_custom == Y_train_ext)/n_train;
    CCR_test = sum(Y_hat_test_custom == Y_test)/n_test;

    CCR_train_svmKern_vec_custom = [CCR_train_svmKern_vec_custom, CCR_train];
    CCR_test_svmKern_vec_custom = [CCR_test_svmKern_vec_custom, CCR_test];
    
    
    %MSE
    X_train_Reconstructed_custom = predict(autoenc_custom,X_train_custom);
    X_test_Reconstructed_custom = predict(autoenc_custom,X_test_custom);
    
    mseError_train_custom = mse(X_train_custom(1:(end - l), :) - X_train_Reconstructed_custom(1:(end - l), :));
    MSE_train_vec_custom = [MSE_train_vec_custom, mseError_train_custom];
    
    mseError_test_custom = mse(X_test_custom(1:(end - l), :) - X_test_Reconstructed_custom(1:(end - l), :));
    MSE_test_vec_custom = [MSE_test_vec_custom, mseError_test_custom];
    
    
    %% Images
    if(k == 1 || k == 11 || k == 31 || k == 51 || k == 101 || k == 151 || k == 201 || k == 251)

        figure(1)
        subplot(4,4,positionCount)
        positionCount = positionCount + 1;
        img = X_test_Reconstructed(1:end, display_num);
        imshow(reshape(img, img_size)) 
        titlevar = {"Original Reconstruction", " k = " + k};
        title(titlevar);

        subplot(4,4,positionCount)
        positionCount = positionCount + 1;
        img = X_test_Reconstructed_custom(1:(end - l), display_num);
        imshow(reshape(img, img_size)) 
        titlevar = {"Modified Reconstruction", " k = " + k};
        title(titlevar);

    end

end

%% Original Image
figure
img = X_test(1:end, display_num);
imshow(reshape(img, img_size)) 
title("Original");


%% Plots
figure()
hold on
plot(k_array(1:end), CCR_train_lda_vec(1:end))
plot(k_array(1:end), CCR_train_lda_vec_custom(1:end))
title_var = {"CCR LDA Train", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
%legend("Train", "Test")
ylim([0.3 1.05])
hold off

figure()
hold on
plot(k_array(1:end), CCR_test_lda_vec(1:end))
plot(k_array(1:end), CCR_test_lda_vec_custom(1:end))
title_var = {"CCR LDA Test:", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
ylim([0.3 1.05])
hold off

figure()
hold on
plot(k_array(1:end), CCR_train_svmLin_vec(1:end))
plot(k_array(1:end), CCR_train_svmLin_vec_custom(1:end))
title_var = {"CCR SVM Linear Train", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
ylim([0.3 1.05])
hold off

figure()
hold on
plot(k_array(1:end), CCR_test_svmLin_vec(1:end))
plot(k_array(1:end), CCR_test_svmLin_vec_custom(1:end))
title_var = {"CCR SVM Linear Test", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
ylim([0.3 1.05])
hold off

figure()
hold on
plot(k_array(1:end), CCR_train_svmKern_vec(1:end))
plot(k_array(1:end), CCR_train_svmKern_vec_custom(1:end))
title_var = {"CCR SVM Kernal Train", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
legend("Train", "Test")
ylim([0.3 1.05])
hold off

figure()
hold on
plot(k_array(1:end), CCR_test_svmKern_vec(1:end))
plot(k_array(1:end), CCR_test_svmKern_vec_custom(1:end))
title_var = {"CCR SVM Kernal Test", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR")
xlabel("k")
legend("Original", "Modified")
ylim([0.3 1.05])
hold off


figure()
hold on
plot(k_array(1:end), MSE_train_vec(1:end))
plot(k_array(1:end), MSE_train_vec_custom(1:end))
title_var = {"Train MSE", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("MSE")
xlabel("k")
legend("Original", "Modified")
hold off


figure()
hold on
plot(k_array(1:end), MSE_test_vec(1:end))
plot(k_array(1:end), MSE_test_vec_custom(1:end))
title_var = {"Test MSE", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("MSE")
xlabel("k")
legend("Original", "Modified")
hold off

figure()
hold on
plot(k_array(1:end), CCR_train_svmKern_vec_custom(1:end) - CCR_train_svmKern_vec(1:end))
%title_var = {"CCR SVM Kernal Train Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
title_var = {"CCR SVM Kernal Train Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR Difference")
xlabel("k")
ylim([-0.2 0.2])
hold off

figure()
hold on
plot(k_array(1:end), CCR_test_svmKern_vec_custom(1:end) - CCR_test_svmKern_vec(1:end))
%title_var = {"CCR SVM Kernal Test Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
title_var = {"CCR SVM Kernal Test Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("CCR Difference")
xlabel("k")
ylim([-0.2 0.2])
hold off


figure()
hold on
plot(k_array(1:end), MSE_train_vec_custom(1:end) - MSE_train_vec(1:end))
%title_var = {"Train MSE Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
title_var = {"Train MSE Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("MSE")
xlabel("k")
hold off

figure()
hold on
plot(k_array(1:end), MSE_test_vec_custom(1:end) - MSE_test_vec(1:end))
%title_var = {"Test MSE difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
title_var = {"Test MSE difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
title(title_var)
ylabel("MSE")
xlabel("k")
hold off

%% Extra plots for LDA and SVM Linear. Commentated out for convienence 
% figure()
% hold on
% plot(k_array(1:end), CCR_train_lda_vec_custom(1:end) - CCR_train_lda_vec(1:end))
% %title_var = {"CCR LDA Train Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title_var = {"CCR LDA Train Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title(title_var)
% ylabel("CCR difference")
% xlabel("k")
% %legend("Train", "Test")
% ylim([-0.2 0.2])
% hold off
% 
% figure()
% hold on
% plot(k_array(1:end), CCR_test_lda_vec_custom(1:end) - CCR_test_lda_vec(1:end))
% %title_var = {"CCR LDA Test Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title_var = {"CCR LDA Train Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title(title_var)
% ylabel("CCR difference")
% xlabel("k")
% ylim([-0.5 0.5])
% hold off
% 
% figure()
% hold on
% plot(k_array(1:end), CCR_train_svmLin_vec_custom(1:end) - CCR_train_svmLin_vec(1:end))
% %title_var = {"CCR SVM Linear Train Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title_var = {"CCR LDA Train Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title(title_var)
% ylabel("CCR Difference")
% xlabel("k")
% ylim([-0.2 0.2])
% hold off
% 
% figure()
% hold on
% plot(k_array(1:end), CCR_test_svmLin_vec_custom(1:end) - CCR_test_svmLin_vec(1:end))
% %title_var = {"CCR SVM Linear Test Difference: " + noi_0 + ", " + noi_1, "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title_var = {"CCR LDA Train Difference: 0-4 vs 5-9", "l = " + l + ", a = " + a + ", epochs = " + epochs};
% title(title_var)
% ylabel("CCR")
% ylabel("CCR Difference")
% ylim([-0.2 0.2])
% hold off





