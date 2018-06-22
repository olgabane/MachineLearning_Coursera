function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%calculate regularized cost function for linear regression
J = 1/(2*m)* sum((X*theta - y).^2) + lambda/(2*m)*sum(theta(2:end, :).^2);

%calculate gradient for j = 0 (unregularized)
grad = (1/m)*((X*theta - y)'* X);
              
%calculate gradient for j = 1:m, including regularization term
theta(1) = 0;
reg = (lambda/m)*(theta);
grad = grad' + reg;
              




% =========================================================================

grad = grad(:);

end
