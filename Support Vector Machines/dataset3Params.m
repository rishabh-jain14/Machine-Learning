function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns our choice of C and sigma for a non-linear dataset
%where we select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns our choice of C and 
%   sigma.
%


C = 1;
sigma = 0.3;

C_list = [0.01 0.03 0.1 0.3 1 3 10 30 100 300];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30 100 300];
result = zeros(length(C_list)*length(sigma_list), 3);
rows = 1;

for i = 1:length(C_list)
  for j = 1:length(sigma_list)
    C_test = C_list(1, i);
    sigma_test = sigma_list(1, j);    
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    predictions = svmPredict(model, Xval);
    prediction_error = mean(double(predictions ~= yval));
    result(rows, :) = [prediction_error, C_test, sigma_test];
    rows = rows+1;
  end
end

sorted_result = sortrows(result, 1);
C = sorted_result(1, 2);
sigma = sorted_result(1, 3);

end
