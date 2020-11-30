function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);
X = [ones(m, 1) X];

p = zeros(size(X, 1), 1);

hidden_attributes1 = sigmoid(X*Theta1');
hidden_attributes1 = [ones(m, 1) hidden_attributes1];

hidden_attributes2 = sigmoid(hidden_attributes1*Theta2');

[~, p] = max(hidden_attributes2, [], 2);

end
