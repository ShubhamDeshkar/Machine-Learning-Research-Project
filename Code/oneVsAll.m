% clc;
% clear all;
% close all;

data = load('illumination.mat');
total_size = 68;

data_set = data.illum;
train_size = 20;

train_data = [];
test_data = [];
for j = 1:total_size % jth subject
    for i = 1:train_size % ith illumination
        train_data = [train_data data_set(:,i,j)];
    end
    test_data = [test_data data_set(:,21,j)];
end

% training here -->

theta = [];
theta0 = [];

for j = 1:train_size:size(train_data,2) % for every subject
    class = [];
    rem_data = train_data;
    for n = j:j+train_size-1 % for every illumination (20)
        % creating a class of 20 illuminations
        class = [class train_data(:,n)];
        rem_data(:,j) = [];
    end
    % X matrix
    X = [class rem_data]';
    % it's labels
    labels = [ones(size(class,2),1); -ones(size(rem_data,2),1)];

    % Gram matrix
    H = (X*X').*(labels*labels');

    % those quadprog conditions
    f = -ones(size(X,1),1);
    B = [labels';zeros(size(X,1)-1,size(X,1))];
    Beq = zeros(size(X,1),1);

    % that in-separable slack parameter condition
    lb = zeros(size(X,1),1);
    ub = 1*ones(size(lb));

    % finally mu's from quadprog
    mu = quadprog(H,f,[],[],B,Beq,lb,ub);

    % theta here
    theta = [theta ((mu.*labels)'*X)'];

    % small trick to get non-zero mu
    [~,index] = max(mu);
    % theta0 here
    theta0 = [theta0 (1/labels(index))-theta(:,end)'*X(index,:)'];
end


acc = 0;
for i = 1:size(test_data,2) % for every testing data
    vect = [];
    for j = 1:size(theta,2)
        value = theta(:,j)'*test_data(:,i)+theta0(j);
        if value > 0
            vect = [vect 1];
        else
            vect = [vect 0];
        end
    end
    
    if sum(vect) == 1
        [~,index] = max(vect);
        if index == i
            disp('correct');
            acc = acc+1;
        end
    else
        disp('incorrect');
    end
end


