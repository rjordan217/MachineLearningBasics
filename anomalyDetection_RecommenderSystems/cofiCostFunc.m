function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

errTerms = R .* (X * Theta' - Y);
thetaReg = lambda * sum(sum(Theta .^ 2));
xReg = lambda * sum(sum(X .^ 2));
J = (sum(sum(errTerms .^ 2)) + thetaReg + xReg) / 2;


X_grad = errTerms * Theta + lambda * X;
Theta_grad = errTerms' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
