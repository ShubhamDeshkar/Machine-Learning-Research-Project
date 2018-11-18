clc;
clear all;
close all;

%the value of number of nearest neighbors.
K = 7;

%this is the total size of the subjects.
total_size = 200;
%this is the size of subjects again, 150 points in each class.
training_size = 150;
data_set = 'data.mat';

%this simple partitions the .mat file into training_size and testing_size.
%But the samples are still interleaved.
training_data = get_training_data(data_set, training_size);
testing_data = get_testing_data(data_set, total_size, training_size);

class_neutral = [];
class_expression = [];
for n = 1: training_size
    %3xn-2 is how neutral faces are indexed.
    class_neutral = [class_neutral training_data(:,3*n-2)];
    %3xn-1 is how expression faces are indexed.
    class_expression = [class_expression training_data(:,3*n-1)];
end

testing_set_N = [];
testing_set_E = [];
for n = 1: total_size-training_size
    %ignoring the illumination class, appending only the neutral and
    %expression class. The illumination class is ignored (discarded).
    testing_set_N = [testing_set_N testing_data(:,3*n-2)];
    testing_set_E = [testing_set_E testing_data(:,3*n-1)];
end
%joining the two test set so 1st half are neutral data points and 2nd half
%are expression data points.
testing_set = [testing_set_N testing_set_E];

%testing without LDA -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%following steps are performed for every test_image.
accuracy = 0;
for n = 1: size(testing_set, 2)
    distance_vector = [];
    if n <= size(testing_set, 2)/2
        true_label = 1;
    else
        true_label = -1;
    end
    %computing L2norm or a testing image to all images in class_neutral
    %with appending label = +1.
    for m = 1: size(class_neutral, 2)
        %distance = L2_norm(testing_set(:,n), class_neutral(:,m));
        distance = norm(testing_set(:,n)-class_neutral(:,m));
        distance_vector = [distance_vector;[distance 1]];
    end
    %computing L2norm or a testing image to all images in class_expression
    %with appending label = -1.
    for m = 1: size(class_expression, 2)
        %distance = L2_norm(testing_set(:,n), class_expression(:,m));
        distance = norm(testing_set(:,n)-class_expression(:,m));
        distance_vector = [distance_vector;[distance -1]];
    end
    %find the computed label using the value of K from distance_vector
    computed_label = get_label(distance_vector, K);
    
    if true_label*computed_label == 1
        accuracy = accuracy + 1;
    end
end
disp('Base acccuracy: ');
disp(accuracy/size(testing_set, 2)*100);

%LDA starts here -->
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%getting theta from LDA function
theta = LDA('data.mat');
accuracy = LDA_KNN(theta,K,class_neutral,class_expression,testing_set);
disp('LDA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);

%PCA starts here --> on 57
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
comp = myPCA([class_neutral class_expression],25);
accuracy = PCA_KNN(comp,K,class_neutral,class_expression,testing_set);
disp('PCA acccuracy: ');
disp((accuracy/size(testing_set,2))*100);


