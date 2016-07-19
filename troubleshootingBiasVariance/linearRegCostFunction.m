function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

thetaForReg = [0; theta(2:end,1)];

lambdaTerm = sum(lambda / (2 * m) * thetaForReg .^ 2);

hYDiff = (X * theta) - y;

J = 1 / (2 * m) * sum((hYDiff) .^ 2) + lambdaTerm;

grad = 1 / m * (hYDiff' * X)' + lambda / m * thetaForReg;

grad = grad(:);

end
