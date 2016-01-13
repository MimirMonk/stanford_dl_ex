function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
% cost
z = [theta'; zeros(1,n)] * X;
ez = exp(z);
ezsum = sum(ez,1);

rows = y;
cols = 1:m;
idx = sub2ind(size(z), rows, cols);
ezj = ez(idx);

J_xy = log(ezj ./ ezsum);
f = -sum(J_xy,2);


% gradient
groundTruth = full(sparse(y, 1:m, 1));
py = ez ./ repmat(ezsum,[num_classes, 1]);
diff_gr = groundTruth - py;
g = -diff_gr * X';
g = g - repmat(g(end,:),[num_classes,1]);
g = g(1:end-1,:)';



  g=g(:); % make gradient a vector for minFunc

