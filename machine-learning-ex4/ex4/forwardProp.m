function [Z2, A2, Z3, H ] = forwardProp( X, Theta1, Theta2 )
%FORWARDPROP Perform forward propagation to compute the hidden layer activations
% and the hypothesis function of a three - layer neural network with weight 
% matrices Theta1 and Theta2. X is an m x n matrix of m n-dimensional examples.
%   Theta1 and Theta2 are matrices of appropriate dimensions for moving
%       from layer 1 (input) to layer 2 (hidden) and layer 2 to layer 3 (output)
%       respectively.

Z2 = Theta1 * X';
A2 = sigmoid(Z2);
A2 = [ones(1, size(A2, 2)); A2];
Z3 = Theta2 * A2;
H = sigmoid(Z3); % k x m matrix of scores for every example

end

