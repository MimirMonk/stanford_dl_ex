function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%

m = size(data,2);
z = cell(numHidden+2, 1);
a = cell(numHidden+2, 1);
z{1} = 0;
a{1} = data;

% MLP
for l = 1:numHidden
    W = stack{l}.W;
    b = stack{l}.b;
    z{l+1} = W * a{l} + repmat(b,[1,m]);
    a{l+1} = activation(z{l+1}, ei.activation_fun);
end

% Softmax
l = numHidden + 1;
W = stack{l}.W;
b = stack{l}.b;
z{l+1} = W * a{l} + repmat(b,[1,m]);
a{l+1} = activation(z{l+1}, 'softmax');

pred_prob = a{end};

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

h = a{end};
rows = labels;
cols = 1:m;
idx = sub2ind(size(h), rows(:), cols(:));
h_idx = h(idx);
h_log = log(h_idx);
cost = -sum(h_log(:))/m;


%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

delta = cell(numHidden+2,1);
% Softmax
groundTruth = full(sparse(rows, cols, 1));
diff = h - groundTruth;
delta{end} = diff;

% MLP
for l = numHidden+1:-1:2
    delta{l} = a{l}.*(1-a{l}) .* (stack{l}.W' * delta{l+1});
end

for l = 1:numHidden+1
    gradStack{l}.W = delta{l+1} * a{l}' / m;
    gradStack{l}.b = sum(delta{l+1}, 2) / m;
end

% delta{end} = a.*(1-a) .* (stack{end}.W' * diff);
% 
% gradStack{end}.W = diff * a{end-1}' / m;
% gradStack{end}.b = sum(diff,2)./m;


%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

cost = cost + ei.lambda / 2 * sum(stack{end}.W(:).^2);
gradStack{end}.W = gradStack{end}.W + ei.lambda * stack{end}.W;


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

function [a] = activation(z, activation_fun)

switch activation_fun
    case 'logistic',
        a = 1./(1+exp(-z));
    case 'softmax',
        ez = exp(z);
        ezsum = sum(ez,1);
        a = ez./repmat(ezsum,[size(z,1),1]);
end

end