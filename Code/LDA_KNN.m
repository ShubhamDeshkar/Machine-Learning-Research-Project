function accuracy = LDA_KNN(theta,K,class1,class2,testing_set)
    
    %projecting both classes on theta
    proj_classNeutr = theta' * class1;
    proj_classExprss = theta' * class2;

    acc = 0;
    for n = 1: size(testing_set, 2)
        distance_vector = [];
        if n <= size(testing_set, 2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        %computing L2norm of a testing image to all images in class_neutral
        %with appending label = +1.
        for m = 1: size(proj_classNeutr, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(theta'*testing_set(:,n)-proj_classNeutr(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
        %computing L2norm of a testing image to all images in class_expression
        %with appending label = -1.
        for m = 1: size(proj_classExprss, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(theta'*testing_set(:,n)-proj_classExprss(:,m));
            distance_vector = [distance_vector;[distance -1]];
        end
        %find the computed label using the value of K from distance_vector
        computed_label = get_label(distance_vector, K);

        if true_label*computed_label == 1
            acc = acc + 1;
        end
    end
    accuracy = acc;
end