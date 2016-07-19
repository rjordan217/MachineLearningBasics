function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

h = sigmoid(X * theta);
lambdaTerm = lambda / (2 * m) * sum(theta(2:end, 1) .^ 2);

J = 1 / m * sum(-y' * log(h) - (ones(m,1) - y)' * log(ones(m,1) - h)) + lambdaTerm;

gradLambda = zeros(size(theta));
gradLambda(2:rows(theta), :) = lambda / m * theta(2:end, 1);
grad = 1 / m * ((h - y)' * X)' + gradLambda;

grad = grad(:);

end
