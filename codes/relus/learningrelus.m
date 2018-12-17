%% Implementation of the noisy-injected stochastic gradient descent proposed in the paper
%  `` Learning ReLU Networks on Linearly Separable Data: Algorithm, Optimality, and Generalization'' by G. Wang, G. B. Giannakis, and J. Chen.

% rng(118)

Params.activations = {'nrelu', 'rrelu'};
orelu = @(x) (x >= 0) .* x;

%% input/architecture dimensions
if isfield(Params, 'd')                     == 0,        Params.d                 = 20;  end             % feature dimension
if isfield(Params, 'k')                     == 0,        Params.k                 = 2;  end     % number of hidden units
if isfield(Params, 'ntrain')          == 0,        Params.ntrain      = 100;  end    	% number of training examples
if isfield(Params, 'ntest')            == 0,        Params.ntest         =  5000;  end		% number of test examples
if isfield(Params, 'vc')                  == 0,        Params.vc                = 1;  end             % constant v values

%% SGD parameters
if isfield(Params, 'T')                    == 0,        Params.T                 = 1000;  end    	% number of SGD epochs
if isfield(Params, 'muu')             == 0,        Params.muu           = 1e-1;  end		% step size / learning parameter
if isfield(Params, 'initvar')        == 0,        Params.initvar      = 0;  end		% nonzero init variance?
if isfield(Params, 'noise')           == 0,        Params.noise        = 10;  end		% noise standard vairance?
if isfield(Params, 'reshuffle')    == 0,        Params.reshuffle  = 1;  end		% SGD over reshuffled data?
if isfield(Params, 'tol')                 == 0,        Params.tol              = 1e-15;  end		% SGD stopping criterion on the loss
if isfield(Params, 'loss')              == 0,        Params.loss           = 1;  end		% compute loss?
if isfield(Params, 'cstep')           == 0,        Params.cstep        = 1;  end		% constant stepsize or not

if isfield(Params, 'reg')                == 0,        Params.reg          = 5e0;  end		% noise standard vairance?
if isfield(Params, 'error')            == 0,        Params.error       = 1;  end		% noise standard vairance?


%% generate linearly separable synthetic data
Xtrain = 1 * rand(Params.ntrain, Params.d);
w = 1 * randn(Params.d, 1);
ytrain = sign(Xtrain * w);
smin = min(ytrain .* (Xtrain * w));
w = w / smin;
ytrain = sign(Xtrain * w);

Xtest =   1 *randn(Params.ntest, Params.d);
ytest = sign(Xtest * w);

% %% reverse labels to get non-linearly separable data
% ind = randi(Params.ntrain, 2, 0);
% ytrain(ind) = -ytrain(ind);
% ind = randi(Params.ntest, 2, 0);
% ytest(ind) = -ytest(ind);

% %% load MNIST training image data
% Params.ntrain = 3000;
% Params.ntest = 1000;
% load('C:\Users\gangwang\Dropbox\multivariate analysis\dPCA\2018ICASSP\used codes\Mnist_data_train.mat');
% % load MNIST training labels data
% load('C:\Users\gangwang\Dropbox\multivariate analysis\dPCA\2018ICASSP\used codes\Mnist_labels_train.mat')
%
% %% take digits 3 and 5 out
% index = (Mnist_labels_train == 3) | (Mnist_labels_train == 5);
% Xall = Mnist_data_train(:, :, index);
% Xdata = reshape(Xall, [size(Xall, 3), size(Xall, 1) * size(Xall, 2)]);
% yall = zeros(length(Mnist_labels_train), 1);
% yall(Mnist_labels_train == 3) = 1;
% yall(Mnist_labels_train == 5) = -1;
% ydata = yall(yall ~= 0);
%
% %% training data
% Xtrain = Xdata(1:Params.ntrain, :);
% ytrain = ydata(1:Params.ntrain, :);
% Xtest = Xdata(Params.ntrain+1 : Params.ntrain+Params.ntest, :);
% ytest = ydata(Params.ntrain+1 : Params.ntrain+Params.ntest);
%
% Params.d = size(Xtrain, 2);



width = 1.2;

for act = 1:length(Params.activations)
    
    act_opt = Params.activations{act};
    
    switch act_opt
        
        case 'nrelu'
            disp('working on relu');
            [f_nrelu, out_nrelu] = nReLU_work(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_nrelu, '-kd', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'lrelu'
            disp('working on leaky relu');
            [f_lrelu, out_lrelu] = lReLU(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_lrelu, '-bv', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'rrelu'
            disp('working on reg_relus');
            [f_reg, out_reg] = nReLU_reg(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_reg, '-m+', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'labs'
            disp('working on labs');
            [f_labs, terror_labs] = Labs(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_labs, '-m+', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'sigmoid'
            [f_sigmoid, terror_sigmoid] = Sigmoid(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_sigmoid, '-rs', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'hsigmoid'
            f_hsigmoid = halfSigmoid(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_hsigmoid, '-g+', 'LineWidth', width, 'MarkerSize', eps);
            end
            
    end
    
end


if Params.loss == 1
    grid
    hx1 = xlabel('Epoch');
    hy1 = ylabel('Training loss');
end

legend('Noisy SGD', 'NSGD Reg');

figure;
plot(out_nrelu.wnorm, 'r');
hold on;
plot(out_reg.wnorm, 'b');
grid
hx1 = xlabel('Iteration');
hy1 = ylabel('W norm');
legend('Noisy SGD', 'NSGD Reg');


% disp([terror_relu, terror_lrelu, terror_labs, terror_sigmoid]);

%%
% width = 1.;
% semilogy(f_relu, '-kd', 'LineWidth', width, 'MarkerSize', .02);
% hold on;
% semilogy(f_lrelu, '-bv', 'LineWidth', width, 'MarkerSize', .02);
% semilogy(plm, '-rs', 'LineWidth', width, 'MarkerSize', 6);
% semilogy(spl, '-m+', 'LineWidth', width, 'MarkerSize', 6);

% figure
% semilogy(f_relu, '--r', 'LineWidth', width);
% hold on;
% semilogy(f_lrelu, '-.b', 'LineWidth', width);
