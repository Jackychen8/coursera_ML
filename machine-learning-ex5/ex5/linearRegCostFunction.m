function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Logistic Regression
%sigmoid function
%h = 1.0 ./ (1.0 + exp(-(X*theta)));
%J = (1/m)*sum(-y'*log(h) - (ones(size(y'))-y')*log(ones(size(h))-h)) + (lambda/(2*m))*sum(theta(2:size(theta),1).^2);
%grad(1) = (1/m) * ((h - y)'*X(:,1))';
%grad(2:size(theta)) = ((1/m) * ((h - y)'*X(:,2:size(theta)))') + (lambda/m)*theta(2:size(theta));

% Linear Regression
h = X*theta;
J = (0.5/m) * sum( ( h - y ).^2 ) + (lambda/(2*m))*sum(theta(2:size(theta)).^2);

grad(1) = (1/m) * sum((h - y).*X(:,1)) ;

grad(2:size(theta)) = (1/m) * ((h - y)'*X(:,2:size(theta)))' + (lambda/m)*theta(2:size(theta));
 
% =========================================================================

grad = grad(:);

end
