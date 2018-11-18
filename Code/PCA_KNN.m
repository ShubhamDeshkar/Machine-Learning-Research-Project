function accuracy = PCA_KNN(comp,K,class1,class2,testing_set)
    classN = comp'*class1;
    classE = comp'*class2;
    
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
        for m = 1: size(classN, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(comp'*testing_set(:,n)-classN(:,m));
            distance_vector = [distance_vector;[distance 1]];
        end
        %computing L2norm of a testing image to all images in class_expression
        %with appending label = -1.
        for m = 1: size(classE, 2)
            %note that testing point is projected onto theta for testing.
            distance = norm(comp'*testing_set(:,n)-classE(:,m));
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