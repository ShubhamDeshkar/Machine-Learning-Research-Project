function theta = LDA(data_set)
    %this is the total size of the subjects.
    total_size = 200;
    %this is the size of subjects again, 150 points in each class.
    training_size = 150;
    %data_set = 'data.mat';

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

    %LDA starts here -->
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %calculating covariance of class1 and class2 (neutral and expression)
    cov_neutral = cov(class_neutral');
    cov_expression = cov(class_expression');

    %adding both the covariances to get 1 final matrix
    cov_matrix = cov_neutral + cov_expression;

    %I tried pseudo-inverse here. Please evaluate.
    inv_cov_matrix = pinv(cov_matrix);

    %calculating individual mean (u) vectors
    mean_neutral = sum(class_neutral, 2)/training_size;
    mean_expression = sum(class_expression, 2)/training_size;

    %adding both means to get final u vector
    final_mean = mean_neutral - mean_expression;

    %now, theta = (sigma^-1)(u)
    theta = inv_cov_matrix * final_mean; 
end