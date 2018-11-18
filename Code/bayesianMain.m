clc;
clear all;
close all;

%this is the total size of the subjects.
total_size = 200;
%this is the size of subjects again, 150 data points in each class.
training_size = 100;
data_set = 'data.mat';

%this simply partitions the .mat file into training_size and testing_size.
%but the samples are still interleaved.
training_data = get_training_data(data_set, training_size);
testing_data = get_testing_data(data_set, total_size, training_size);

class_neutral = [];
class_expression = [];
for n = 1: training_size
    %3*n-2 is how neutral faces are indexed.
    class_neutral = [class_neutral training_data(:,3*n-2)];
    %3*n-1 is how expression faces are indexed.
    class_expression = [class_expression training_data(:,3*n-1)];
end

%partitioning testing_set as well so it is easier to determine
%accuracy
testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    %ignoring the illumination class, appending only the neutral and
    %expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
%joining the two test set so 1st half are neutral points and 2nd half are
%expression points.
testing_set = [testing_set_N testing_set_E];

%mean of both the classes are calculated according to the ML estimation
mean_neutral = sum(class_neutral,2)/size(class_neutral,2);
mean_expression = sum(class_expression,2)/size(class_expression,2);

cov_neutral = cov(class_neutral');
cov_expression = cov(class_expression');

%check this with someone
%regularizing the matrix to find det(covariance matrix).
I = eye(size(cov_neutral));
noise = 0.4*I;
cov_neutral = cov_neutral + noise;
cov_expression = cov_expression + noise;

%inverting covariance matrices using pseudo-inverse since the matrices
%generated are singular.
inv_cov_neutral = pinv(cov_neutral);
inv_cov_expression = pinv(cov_expression);

%testing starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
accuracy = 0;
for n = 1:size(testing_set,2)
    if n <= size(testing_set,2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    %creating model for class_neutral and class_expression
    P_neutral = (1/sqrt(2*pi*det(cov_neutral)))*exp(-0.5*(testing_set(:,n)-mean_neutral)'*inv_cov_neutral*(testing_set(:,n)-mean_neutral));
    P_expression = (1/sqrt(2*pi*det(cov_expression)))*exp(-0.5*(testing_set(:,n)-mean_expression)'*inv_cov_expression*(testing_set(:,n)-mean_expression));
    
    %appending labels to posteriors: +1 to neutral and -1 to expression class
    posteriors = [P_neutral 1;P_expression -1];
    %finding max of the two posterior probabilities
    [~,index] = max(posteriors(:,1));
    
    %proper labelling for comparison
    if index == 1
        computed_label = 1;
    elseif index == 2
        computed_label = -1;
    end
    
    %comparison of labels
    if true_label*computed_label == 1
        accuracy = accuracy+1;
    end
end
disp('Base acccuracy: ');
disp((accuracy/size(testing_set,2))*100);


%LDA starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
theta = LDA('data.mat');
accuracy = LDA_bayesian(theta,class_neutral,class_expression,testing_set);
disp('LDA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);


%PCA here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
comp = myPCA([class_neutral class_expression],73);
accuracy = PCA_bayesian(comp,class_neutral,class_expression,testing_set);
disp('PCA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);




