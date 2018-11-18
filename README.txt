-----------------------------------------------------------------
-----------------------------------------------------------------
ENEE633 - STATISTICAL PATTERN RECOGNITION | FALL-2018
PROJECT: 1
-----------------------------------------------------------------
-----------------------------------------------------------------


Majority of the tasks for the purposes of this project are carried out on the data set called data which is in data.mat

To test the Bayesian classification task for neutral vs. facial expression classification, please run the bayesianMain.m file. This matlab code makes the use of get_training_data.m, get_testing_data.m.
Also, note that in order to test the same classification task after transofrmations like LDA and PCA, no seperate file is required. The bayesianMain.m file provides the evaluation after LDA and PCA as well. In this case, it makes the use of function written in LDA.m, LDA_bayesian.m, myPCA.m and PCA_bayesian.m files.

-----------------------------------------------------------------
Follow the exact same procedure for testing K - nearest neighbors algorithm. It also depends upon get_training_data.m, get_testing_data.m, LDA.m, myPCA.m. Additionally, it depends upon LDA_KNN.m and PCA_KNN.m

-----------------------------------------------------------------
To test the SVM, use the file SVMmain.m. It depends on the function from get_training_data.m, get_testing_data.m and SVMtesting.m for evaluation.
Additionally, for RBF and Polynomial kernel, it does kernelization using Kernelsvm.m and does testing using kernelTesting.m

-----------------------------------------------------------------
For Boosted SVM, use the file. It again uses the get_training_data.m and get_testing_data.m to construct a weak classifier and SVMtesting.m for evaluation and finally uses the boostedTesting.m for testing the combined classifiers.

-----------------------------------------------------------------
Moreover, subject classification was performed using Multiclass SVM on Illumination data set. Use oneVsAll.m for testing it. It does not depend on any other functions but only on the data set itself. Later, multiclass subject classification is performed using KNN in multiClassKNN.m. This file is also independent of any function but only depend on pose.mat and illumination.mat. Only a few modifications are needed to run the code with these two data sets.

-----------------------------------------------------------------

