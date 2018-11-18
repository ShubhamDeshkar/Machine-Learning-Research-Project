clc;
clear all;
close all;

% this is the total size of the subjects.
total_size = 200;
% this is the size of subjects again, 150 data points in each class.
training_size = 100;
data_set = 'data.mat';

% this simply partitions the .mat file into training_size and testing_size.
% but the samples are still interleaved.
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

% partitioning testing_set as well so it is easier to determine
% accuracy
testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    % ignoring the illumination class, appending only the neutral and
    % expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
% joining the two test set so 1st half are neutral points and 2nd half are
% expression points.
testing_set = [testing_set_N testing_set_E];


% SVM using QuadProg starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% concatenating both the classes in the following way
X = [class_neutral class_expression]';

% the slack parameter C is chosen here
C = 0.27;

% creating label vectors: positive and negative and then concatenating them
% into column vectors
pos_label = ones(1,size(class_neutral,2))';
neg_label = -ones(1,size(class_expression,2))';
labels = [pos_label;neg_label];

% creating labelled Gram Matrix
H = (X*X').*(labels*labels');

% modelling parameters according to the ones given in the documentation of
% QuadProg.
f = -ones(size(X,1),1);
B = [labels';zeros(size(X,1)-1,size(X,1))];
Beq = zeros(size(X,1),1);

% additional condition for non-separable classes here. The lower bound is
% 0, and the upperbound is C
lb = zeros(size(X,1),1);
ub = C*ones(size(lb));

% solving the minimization problem using the 'quadprog' function in MATLAB.
mu = quadprog(H,f,[],[],B,Beq,lb,ub);

% the values of mu obtained are not absolutely zero, but they are not
% comparable to the bigger values as well. Hence after a careful
% observation, the threshold is set to 10e-8 and all the mu's below the
% threshold are reduced to zeros.
mu_ = [];
for i = 1:size(mu,1)
    if mu(i) <= 10^-8
        mu_(i) = 0;
    else
        mu_(i) = mu(i);
    end
end
% mu_ is an appropriate vector with small values reduced to zeros.
mu_ = mu_';

% obtaining the values of wt. vector and bias term for linear
% classification
theta = ((mu_.*labels)'*X)';

% looking for random index of non-zero mu value (support-vector) to use it
% to find the bias (theta0)
[~,index] = max(mu_);
theta0 = (1/labels(index)) - theta'*X(index,:)';

accuracy = SVMtesting(theta,theta0,testing_set);
disp('base accuracy:');
disp(accuracy);

r = 0.3;
Ck = 10000;
[thetaPOLY,theta0POLY] = Kernelsvm('poly',r,Ck,class_neutral,class_expression);
accuracyPOLY = kernelTesting('poly',r,thetaPOLY,theta0POLY,testing_set);
disp('POLY accuracy:');
disp(accuracyPOLY);

sigmaSq = 100;
Ck = 2;
[thetaRBF,theta0RBF] = Kernelsvm('rbf',sigmaSq,Ck,class_neutral,class_expression);
accuracyRBF = kernelTesting('rbf',sigmaSq,thetaRBF,theta0RBF,testing_set);
disp('RBF accuracy:');
disp(accuracyRBF);

