function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

X = [ones(m, 1) X]';


y_matrix = zeros(m, num_labels);

for c = 1:m
y_matrix(c, y(c)) = 1;
end

z2 = Theta1 * X;
a2 = [ones(1,m); sigmoid(z2)];

z3 = Theta2 * a2;
a3 = sigmoid(z3);

pre_reg = 0;
for i = 1:m
  for j = 1:num_labels
    a_val = a3(j, i);
    y_val = y_matrix(i,j);
    pre_reg += log(a_val) * (-y_val) - log(1 - a_val) * (1 - y_val);
  end
end

theta1_sqsum = sum(sum(Theta1(:, 2:end) .^ 2));
theta2_sqsum = sum(sum(Theta2(:, 2:end) .^ 2));
lambda_term = lambda / (2 * m) * (theta1_sqsum + theta2_sqsum);

J = 1 / m * pre_reg + lambda_term;


del3 = a3' - y_matrix;
del2 = ((del3 * Theta2)'(2:end, :)) .* sigmoidGradient(z2);

grad1_lambda = lambda / m * Theta1;
grad1_lambda(1:end, 1) = 0;

grad2_lambda = lambda / m * Theta2;
grad2_lambda(1:end, 1) = 0;

Theta1_grad = 1 / m * del2 * X' + grad1_lambda;
Theta2_grad = 1 / m * (a2 * del3)' + grad2_lambda;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
