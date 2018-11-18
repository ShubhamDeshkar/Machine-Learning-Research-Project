function something = myPCA(training_data,number)
    coeff = pca(training_data');
    coeff = coeff(:,1:number);
    something = coeff;
end
