function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

	for iter = 1:num_iters

		% ====================== YOUR CODE HERE ======================
		% Instructions: Perform a single gradient step on the parameter vector
		%               theta. 
		%
		% Hint: While debugging, it can be useful to print out the values
		%       of the cost function (computeCost) and gradient here.
		%

		%compute Sums
		S1=0;
		S2=0;
		for i=1:m,
			prediction=theta'*[X(i,1); X(i,2)];
			predictionDeviation = (prediction - y(i));

			S1=S1 + predictionDeviation*X(i,1);
			S2=S2 + predictionDeviation*X(i,2);

		end;

		S=[S1;S2];
		
		%fprintf('S1 = %f\n', S1);
		%fprintf('S2 = %f\n', S2);
		
		%theta(1)=theta(1)-alpha*(1/m)*S1;
		%theta(2)=theta(2)-alpha*(1/m)*S2;
		theta=theta-alpha/m*S;

		%fprintf('theta(1) = %f\n', theta(1));
		%fprintf('theta(2) = %f\n', theta(2));
		
		% Save the cost J in every iteration    
		J_history(iter) = computeCost(X, y, theta);

		%fprintf('J_history(%f) = %f\n', iter, J_history(iter));
		
	end

end
