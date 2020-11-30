function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y);
J = 0;
grad = zeros(size(theta));
predictions = sigmoid(X*theta);
J = (1/m)*sum(-y.*log(predictions) .- (1 - y).*log(1-predictions)) + (lambda/(2*m))*sum(theta(2:size(theta)(1)).^2)
temp = theta;
temp(1) = 0;
grad = ((1/m) * X' * (predictions-y))  + ((lambda/m) .* temp)
grad = grad(:);

end
