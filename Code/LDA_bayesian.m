function accuracy = LDA_bayesian(theta,class1,class2,testing_set)
    %just re-defining names
    classNeutr = class1;
    classExprss = class2;
    training_size = size(class1,2);
    
    %projecting both classes on theta
    proj_classNeutr = theta' * classNeutr;
    proj_classExprss = theta' * classExprss;

    %finding mean of the two classes
    mean_Neutr = sum(proj_classNeutr)/training_size;
    mean_Exprss = sum(proj_classExprss)/training_size;

    %finding the covariance & inv(covariacne) of both classes
    cov_neutr = cov(proj_classNeutr');
    cov_exprss = cov(proj_classExprss');

    inv_cov_neutr = inv(cov_neutr);
    inv_cov_exprss = inv(cov_exprss);

    %finding determinant of two classes.
    det_cov_neutr = det(cov_neutr);
    det_cov_exprss = det(cov_exprss);

    %testing starts here -->
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    acc = 0;
    for n = 1:size(testing_set,2)
        if n <= size(testing_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        %creating model for class_neutral and class_expression
        P_Neutr = (1/sqrt(2*pi*det_cov_neutr))*exp(-0.5*(theta'*testing_set(:,n)-mean_Neutr)'*inv_cov_neutr*(theta'*testing_set(:,n)-mean_Neutr));
        P_Exprss = (1/sqrt(2*pi*det_cov_exprss))*exp(-0.5*(theta'*testing_set(:,n)-mean_Exprss)'*inv_cov_exprss*(theta'*testing_set(:,n)-mean_Exprss));

        %appending labels to posteriors: +1 to neutral and -1 to expression class
        post = [P_Neutr 1;P_Exprss -1];
        %finding max of the two posterior probabilities
        [~,index] = max(post(:,1));

        %proper labelling for comparison
        if index == 1
            computed_label = 1;
        elseif index == 2
            computed_label = -1;
        end

        %comparison of labels
        if true_label*computed_label == 1
            acc = acc+1;
        end
    end
    accuracy = acc;
end