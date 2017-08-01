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

diff = X * theta - y;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J = (norm(diff) ^ 2 + ...
    lambda * norm(theta(2 : end)) ^ 2) / (2 * m);

theta0_grad = mean(diff .* X(:, 1));
theta1_grad = (X(:, 2 : end)' * diff + lambda * theta(2 : end)) / m;

grad = [theta0_grad; theta1_grad];


% =========================================================================

grad = grad(:);

end
