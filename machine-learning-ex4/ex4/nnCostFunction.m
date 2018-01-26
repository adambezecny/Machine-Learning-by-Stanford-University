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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X=[ones(size(X,1),1) X];%add bias unit
z2 = Theta1 * X';
a2=sigmoid(z2);
a2=a2';
a2=[ones(size(a2,1),1) a2];%add bias unit
z3=Theta2 * a2';
a3=sigmoid(z3);
a3=a3';

S=0;
identity_matrix = eye(num_labels);
y_vectorized=zeros(m, num_labels);

for i=1:m,
    y_vectorized(i,:)=identity_matrix(:, y(i))';
end;

%for i=1:m,
%    for k=1:num_labels,
%        S = S + (y_vectorized(i,k) * log(a3(i,k)) + (1 - y_vectorized(i,k)) * log(1 - a3(i,k)) );
%    end;
%end;

S = sum(sum((y_vectorized .* log(a3) + (1 - y_vectorized) .* log(1 - a3) )));

%regularization
reg_Theta1 = (Theta1(:,2:size(Theta1,2))) .^ 2;
reg_Theta2 = (Theta2(:,2:size(Theta2,2))) .^ 2;

%cost function without regularization
%J=-S/m;

%regularized cost function
J=-S/m + (lambda/(2*m)) * (sum(sum(reg_Theta1)) + sum(sum(reg_Theta2)));

%gradient computation

for i=1:m,
    
        a1 = X(i,:);
        z2 = Theta1 * a1';
        a2 = sigmoid(z2);
        a2 = a2';
        
        a2 = [ones(size(a2,1),1) a2];%add bias unit
        z3 = Theta2 * a2';
        a3 = sigmoid(z3);
        a3 = a3';
        
        d3 = (a3 - y_vectorized(i,:))'; 
        
        %discard change for bias unit
        tmp1 = Theta2' * d3;
        tmp1 = tmp1(2:size(tmp1,1),:);
        d2 = tmp1 .* sigmoidGradient(z2);
        
        Theta1_grad = Theta1_grad + d2*a1;
        Theta2_grad = Theta2_grad + d3*a2;
        
end;

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

%create regularized theta  matrix from original theta matrices leaving out
%first column which represents weights of bias units. multiply non-bias
%units with regualrization expression lambda/m
Theta1_grad_reg = lambda/m*ones(size(Theta1,1), size(Theta1,2)-1);
Theta2_grad_reg = lambda/m*ones(size(Theta2,1), size(Theta2,2)-1);

%add first column with zeros for non-regualrized bias units
Theta1_grad_reg = [zeros(size(Theta1,1), 1) Theta1_grad_reg];
Theta2_grad_reg = [zeros(size(Theta2,1), 1) Theta2_grad_reg];

%add regularization to theta metrices in element wise manner
Theta1_grad = Theta1_grad + Theta1 .* Theta1_grad_reg;
Theta2_grad = Theta2_grad + Theta2 .* Theta2_grad_reg;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
