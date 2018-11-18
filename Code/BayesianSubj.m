clc;
clear all;
close all;

% this is the total size of the subjects.
total_size = 200;
% this is the size of subjects again, 100 data points in each class.
training_size = 170;
data_set = 'data.mat';

% training_data will contain interleaved images of neutral and expression
% for the whole data_set
% now, the neutral image is indexed by training_data(:,2*n-1) and the
% facial expression image is indexed by training_data(:,2*n)
training_data = get_subject_train(data_set,total_size);
% testing_data will contain all the 200 illumination variation. For subject
% 'n', the training point is indexed testing_data(:,n) as well.
testing_data = get_subject_test(data_set,total_size);

covar = [];
mean = [];
post_vect = [];
acc = 0;
for n = 1:total_size    
    for m = 1:total_size
        % constructing the class using only the neutral face and expression
        % face images
        class = [training_data(:,2*m-1) training_data(:,2*m)];

        % calculating the mean
        mean = sum(class,2)/size(class,2);
        
        % calculating the covariance of the class
        covar = cov(class');

        % regulrization for removing non-technical singularity problem
        noise = 0.7*eye(length(class),length(class));
        % covariance of the class:
        covar = covar+noise;
        % inverting covariance matrix using pseudo-inverse (SVD)
        inv_covar = pinv(covar);
        
        posterior = (1/sqrt(2*pi*det(covar)))*exp(-0.5*(testing_data(:,n)-mean)'*inv_covar*(testing_data(:,n)-mean));
        post_vect = [post_vect;posterior];
    end
    [~,index] = max(post_vect);
    if index == n
        acc = acc + 1;
        fprintf('%d. correct\n',n);
    else
        fprintf('%d. incorrect\n',n);
    end
    post_vect = [];
end
disp('Accuracy:');
disp(acc/size(testing_data,2));