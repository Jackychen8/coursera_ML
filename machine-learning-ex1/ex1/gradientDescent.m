function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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

    % hypothesis: transpose(theta)*X
    % parameters: theta
    % Cost function: J
    
    % Gradient Descent: theta = theta - alpha*derivative of J with respect to theta
    % Gradient Descent: theta = theta - (alpha/m)*sum();
    % Gradient Descent for one theta: theta1 = theta1 - (alpha/m)*sum(); 
    
    % (X*theta - y)'*X produces an mx2 matrix
    % sum reduces this to 1 number which is incorrect
    % must sum up each side separately
    
    %dJ = sum((X*theta - y)'*X); turns it into 1 number
    dJ = (X*theta - y)'*X;
    %print(dJ);
    
    % the key was that both X1 and X2 need to be multiplied by the whole thing
    %dJ0 = sum( (X*theta - y)'*X(:,1) );
    %dJ1 = sum( (X*theta - y)'*X(:,2) );
    theta = theta - (alpha/m)*[sum(dJ(:,1));sum(dJ(:,2))];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
    %print("Cost:", computeCost(X, y, theta), "\ntheta: ", theta  );
end

end
