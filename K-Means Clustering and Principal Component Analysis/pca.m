function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

covariance_matrix = (1/m)*X'*X;

[U, S, V] = svd(covariance_matrix);




% =========================================================================

end
