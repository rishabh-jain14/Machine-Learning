function [J, Z] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

predictions = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(predictions) .- (1 - y).*log(1-predictions)) + (lambda/(2*m))*sum(theta(2:size(theta)(1)).^2)
grad = (1/m).* X(:, 1)'*(predictions-y)
Z = [grad; ((1/m) * (X(:, 2:size(X)(2))' * (predictions-y)))  + ((lambda/m) .* theta(2:size(theta)(1)))]
% =============================================================

end
