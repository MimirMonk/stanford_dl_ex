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

% MLP
numCases = size(data,2);
inputData = cell(numHidden+1,1);
inputData{1} = data;
for l = 1:numHidden
    W = stack{l}.W;
    b = stack{l}.b;
    z = W * inputData{l} + repmat(b,[1,numCases]);
    a = activation(z, ei.activation_fun);
    hAct{l} = a;
    inputData{l+1} = a;
end

% Softmax
l = numHidden+1;
W = stack{l}.W;
b = stack{l}.b;
z = W * inputData{l} + repmat(b,[1,numCases]);
a = activation(z, 'softmax');
hAct{l} = a;


%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];
  % prediction
  [~,pred_prob] = max(a,[],1);
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%

a = hAct{end};
rows = labels;
cols = (1:numCases)';
idx = sub2ind(size(a), rows, cols);
aj = a(idx);
J_xy = log(aj);
cost = -sum(J_xy,1)/numCases;

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%

deltaStack = cell(numHidden+1,1);

% Softmax
groundTruth = full(sparse(labels, 1:numCases, 1));
diff_gr = groundTruth - a;
deltaStack{l} = -diff_gr;
gradStack{l}.W = (deltaStack{l} * inputData{l}')./numCases + stack{l}.W * ei.lambda;
gradStack{l}.b = sum(deltaStack{l},2)./numCases;

% MLP
for l = numHidden:-1:1
    deltaStack{l} = inputData{l+1}.*(1-inputData{l+1}) .* (stack{l+1}.W'*deltaStack{l+1});
    gradStack{l}.W = (deltaStack{l} * inputData{l}')./numCases + stack{l}.W * ei.lambda;
    gradStack{l}.b = sum(deltaStack{l},2)./numCases;
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end


function a = activation(z,type)
switch type
    case 'logistic'
        a = 1./(1+exp(-z));
    case 'softmax'
        ez = exp(z);
        ezsum = sum(ez,1);
        a = ez./repmat(ezsum,[size(ez,1),1]);
end
end


