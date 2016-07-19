function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular,
%       it returns two vectors of the same length - error_train and
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).


% Number of training examples
m = size(X, 1);

error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for c=1:m
current_sample = X(1:c, :);
current_y = y(1:c);
[theta] = trainLinearReg(current_sample, current_y, lambda);
[error_train(c), _] = linearRegCostFunction(current_sample, current_y, theta, 0);
[error_val(c), _]  = linearRegCostFunction(Xval, yval, theta, 0);
end

end
