function accuracy = kernelTesting(kernel,param,theta,theta0,testing_set)
    acc = 0;
    % kernel transformation starts here: according to the name of the
    % kernel and the parameter for the given kernel. (sigmaSq or r)
    K = [];
    for n = 1:size(testing_set,2)
        for m = 1:size(testing_set,2)
            if strcmp(kernel,'rbf')
                % radial basis function kernel
                K(n,m) = exp(-((norm(testing_set(:,n)-testing_set(:,m)))^2)/(param));
            elseif strcmp(kernel,'poly')
                % ploynomial kernel
                K(n,m) = (testing_set(:,n)'*testing_set(:,m)+1)^param;
            end
        end
    end

    % actual testing begins here -->
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1:size(K,2)
        % asssigning true labels: +1 for upto 1st half of the testing_set
        % and -1 for the 2nd half of the testing_set
        if i <= size(testing_set,2)/2
            true_label = 1;
        else
            true_label = -1;
        end
        
        % using the test image in the linear predictor
        value = theta'*K(:,i) + theta0;
        % multiplying value with the true_label for easy comparison to
        % evaluate accuracy
        prediction = value*true_label;
        
        % self-explainatory
        if prediction > 0
            acc = acc+1;
        end
    end
    accuracy = (acc/size(K,2))*100;
end