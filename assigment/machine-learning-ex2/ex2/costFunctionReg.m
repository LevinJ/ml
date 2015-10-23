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


g_X_theta = sigmoid (X * theta);

J1 = log(g_X_theta') * y + log((1 - g_X_theta)') * (1 -y);

J1 = -J1 / m;

J2 = (theta' * theta)- theta(1)*theta(1);
J2 = lambda * J2/(2*m);

J = J1 + J2;

grad1 = (X' * (g_X_theta - y))/m;
grad2 = lambda * theta / m;
grad = grad1 + grad2;
grad(1) = grad1(1);





% =============================================================

end
