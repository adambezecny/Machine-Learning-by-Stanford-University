function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


S=0;
predictions=zeros(m,1);
n = size(theta);

for i=1:m,
	predictionH=sigmoid(theta'*X(i,:)');
    predictions(i)=predictionH;
    S = S + (-1*y(i)*log(predictionH) - (1 - y(i))*log(1 - predictionH));
end;

S_theta_squares=0;

%start from two, do not penalize theta0 (i.e. theta(1) since we are indexing from 1 in matlab)
for i=2:n,
    S_theta_squares = S_theta_squares +theta(i)^2;
end;

J=1/m*S + lambda/(2*m)*S_theta_squares;

for j=1:size(theta),
    for i=1:m,
        grad(j)=grad(j) + (predictions(i)-y(i))*X(i,j);
    end;
    if j==1
        grad(j)=grad(j)*1/m;
    else
        grad(j)=grad(j)*1/m+lambda/m*theta(j);
    end;    
end;    


% =============================================================

end
