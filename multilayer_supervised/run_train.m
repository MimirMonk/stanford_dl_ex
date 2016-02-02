% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function

%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];

% add common directory to your path for
% minfunc and mnist data helpers
addpath ../common;
addpath(genpath('../common/minFunc_2012/minFunc'));

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [256, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 0;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'logistic';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);




%% gradient check
% gradCheck = false;
% if gradCheck
%     data_train = data_train(:,1:30);
%     labels_train = labels_train(1:30);
%     ei.layer_sizes = [20, ei.output_dim];
%     stack = initialize_weights(ei);
%     stack{end}.W = stack{end}.W - repmat(stack{end}.W(end,:),[size(stack{end}.W,1),1]);
%     params = stack2params(stack);
% 
%     [cost, grad] = supervised_dnn_cost(params, ei, data_train, labels_train);
%     epsilon = 10^-4;
%     len = numel(params);
%     numgrad = zeros(len,1);
%     for i=1:len
%         disp(i)
%         q = zeros(len,1);
%         q(i) = epsilon;
%         params_plus = params + q;
%         params_minus = params - q;
%         [costplus, ~] = supervised_dnn_cost(params_plus, ei, data_train, labels_train);
%         [costminus, ~] = supervised_dnn_cost(params_minus, ei, data_train, labels_train);
%         numgrad(i) = ( costplus-costminus ) / 2 / epsilon;
%     end
%     diff = norm(numgrad-grad)/norm(numgrad+grad);
%     disp(diff); 
% end






%% setup minfunc options
options = [];
options.display = 'iter';
options.maxFunEvals = 1e6;
options.Method = 'lbfgs';

%% run training
[opt_params,opt_value,exitflag,output] = minFunc(@supervised_dnn_cost,...
    params,options,ei, data_train, labels_train);

%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
[~,pred] = max(pred);
acc_test = mean(pred'==labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
[~,pred] = max(pred);
acc_train = mean(pred'==labels_train);
fprintf('train accuracy: %f\n', acc_train);
