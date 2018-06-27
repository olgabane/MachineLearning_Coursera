function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%vectors of C, sigma values to test
C_vect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma_vect = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_mat = 0;

%use nested for loop to test all possible combinations of C, sigma values
for i = 1:length(C_vect)
    for j = 1:length(sigma_vect)
        C = C_vect(i);
        sigma = sigma_vect(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        %error matrix
        error_mat(i, j) = mean(double(predictions ~= yval));
    end
end

%get C, sigma values that minimize error
[minval, row] = min(min(error_mat,[],2))
[minval, col] = min(min(error_mat,[],1))
C = C_vect(row);
sigma = sigma_vect(col);


% =========================================================================

end
