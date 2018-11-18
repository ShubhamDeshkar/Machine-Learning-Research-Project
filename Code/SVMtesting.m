function accuracy = SVMtesting(theta,theta0,testing_set)
    acc = 0;
    for i = 1:size(testing_set,2)
        % asssigning true labels: +1 for upto 1st half of the testing_set
        % and -1 for the 2nd half of the testing_set
        if i <= size(testing_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        % using the test image in the linear predictor
        value = theta'*testing_set(:,i) + theta0;
        % multiplying value with the true_label for easy comparison to
        % evaluate accuracy
        prediction = value*true_label;
        
        % self-explainatory
        if prediction > 0
            acc = acc+1;
        end
    end
    accuracy = (acc/size(testing_set,2))*100;
end





