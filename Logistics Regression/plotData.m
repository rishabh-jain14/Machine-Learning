function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

plot(X(y==1, 1), X(y==1, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 10);
plot(X(y==0, 1), X(y==0, 2), 'ko', 'MarkerFaceColor', 'r' , 'MarkerSize', 10);

hold off;

end
