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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %25x401
Theta2_grad = zeros(size(Theta2)); %10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


X = [ones(m, 1) X];

Z2 = Theta1 * X'; %25x5000

A2 = sigmoid(Z2); %25x5000

A2 = [ones(m,1) A2']; % 5000x26

Z3 = Theta2 * A2'; % 10x5000

H_theta = sigmoid(Z3);

% size(y) x 10 vector 
Y_new = zeros(num_labels, m); % 10*5000
for i=1:m,
  Y_new(y(i),i)=1;
end

J = (1/m) * sum ( sum ( (-Y_new) .* log(H_theta) - (1-Y_new) .* log(1-H_theta) ));

%% REGULARIZATION

t1 = Theta1(:, 2:size(Theta1,2));
t2 = Theta2(:, 2:size(Theta2,2));

factor_reg = lambda / (2*m) * (sum( sum ( t1.^ 2 )) + sum( sum ( t2.^ 2 )));

J = J + factor_reg;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a %               binary vector of 1's and 0's to be used with the neural network %               cost function.  %
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
 for t = 1:m 
    A1 = X(t, :)'; % 401x1

    Z2 = Theta1 * A1; % 25x1
    H1 = sigmoid(Theta1 * A1); %25x1

    A2 = [1; H1]; % 26 x 1

    Z3 = Theta2 * A2;
    A3 = sigmoid(Z3); %10x1

% Step 2: calculate diff between expected and actual answer

  delta3 = A3 - Y_new(:,t); %10x1

% Step 3
  Z2 = [1 ; Z2]; %26x1

  delta2 = (Theta2' * delta3) .* sigmoidGradient(Z2); % 26x1

% Step 4 
  delta2 = delta2(2:end);


  Theta2_grad = Theta2_grad + delta3*A2'; 

  Theta1_grad = Theta1_grad + delta2*A1';

 endfor 

  Theta2_grad = (1/m) * Theta2_grad; % (10*26)
  Theta1_grad = (1/m) * Theta1_grad; % (25*401)
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda/m)*Theta1(:, 2:end)
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda/m)*Theta2(:, 2:end)


% =========================================================================

% Unroll gradients
 grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
