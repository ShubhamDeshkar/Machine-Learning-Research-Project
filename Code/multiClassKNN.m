clc;
clear all;
close all;

data = load('pose.mat');
total_size = 68;
train_size = 13;

K = input('Enter the value of K: ');

% ith pose of jth subject (:,:,i,j)

all_data = [];
all_labels = [];
for j = 1:total_size
    for i = 1:train_size
        img = data.pose(:,:,i,j);
        all_data = [all_data img(:)];
    end
    label = j*ones(1,train_size);
    all_labels = [all_labels label];
end

% test_data = [];
% test_labels = [];
% for j = 1:total_size
%     for i = train_size+1:13
%         img = data.pose(:,:,i,j);
%         test_data = [test_data img(:)];
%     end
%     label = j*ones(1,13-train_size);
%     test_labels = [test_labels label];
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

acc = 0;
for n = 1:size(all_data,2)-1 % for all the testing_data
    dist_vect = [];
    test_data = all_data;
    test_labels = all_labels;
    test_data(:,n) = [];
    test_labels(:,n) = [];
    for m = 1:size(test_data,2) % for all the training_data
        dist = norm(test_data(:,m)-all_data(:,n));
        dist_vect = [dist_vect dist];
    end
    dist_vect = [dist_vect; test_labels];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    subj_number = [];
    subj_index = [];
    vote_vect = zeros(1,total_size);
    for k = 1:K
        [~,index] = min(dist_vect(1,:));
        subj_index = dist_vect(2,index);
        vote_vect(subj_index) = vote_vect(subj_index) + 1;
        subj_number = [subj_number dist_vect(2,index)];
        dist_vect(:,index) = [];
    end
    
    [~,index] = max(vote_vect);
    if index == all_labels(n)
        disp('correct');
        acc = acc + 1;
    else
        disp('incorrect');
    end
end

disp(acc/size(all_data,2));
