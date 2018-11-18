function accuracy = PCA_bayesian(comp,class1,class2,testing_set)
    classN = comp'*class1;
    classE = comp'*class2;
    %training_size = size(class1,2);
    
    meanN = sum(classN,2)/size(classN,2);
    meanE = sum(classE,2)/size(classE,2);
    
    covN = cov(classN');
    covE = cov(classE');
    
    invCovN = pinv(covN);
    invCovE = pinv(covE);
    
    %regularization
    I = eye(size(covN));
    noise = 0.001*I;
    covN = covN + noise;
    covE = covE + noise;
    
    detN = det(covN);
    detE = det(covE);
    
    %testing starts here -->
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    acc = 0;
    for n = 1:size(testing_set,2)
        if n <= size(testing_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        %creating model for class_neutral and class_expression
        P_Neutr = (1/sqrt(2*pi*detN))*exp(-0.5*((comp'*testing_set(:,n)-meanN))'*invCovN*(comp'*testing_set(:,n)-meanN));
        P_Exprss = (1/sqrt(2*pi*detE))*exp(-0.5*((comp'*testing_set(:,n)-meanE))'*invCovE*(comp'*testing_set(:,n)-meanE));

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