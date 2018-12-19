%% Implementation of the noisy-injected stochastic gradient descent proposed in the paper
%  `` Learning ReLU Networks on Linearly Separable Data: Algorithm, Optimality, and Generalization'' 
%   by G. Wang, G. B. Giannakis, and J. Chen, submitted to IEEE Trans. on  Signal Process. on Dec. 2018

% rng(118)
close all

Params.activations = {'nrelu'};
orelu = @(x) (x >= 0) .* x;

%% input/architecture dimensions
if isfield(Params, 'd')                     == 0,        Params.d                 = 128;  end             % feature dimension
if isfield(Params, 'k')                     == 0,        Params.k                 = 2;  end     % number of hidden units
if isfield(Params, 'ntrain')          == 0,        Params.ntrain      = 100;  end    	% number of training examples
if isfield(Params, 'ntest')            == 0,        Params.ntest         =  100;  end		% number of test examples
if isfield(Params, 'vc')                  == 0,        Params.vc                = 1;  end             % constant v values

%% SGD parameters
if isfield(Params, 'T')                    == 0,        Params.T                 = 500;  end    	% number of SGD epochs
if isfield(Params, 'muu')             == 0,        Params.muu           = 1e-1;  end		% step size / learning parameter
if isfield(Params, 'initvar')        == 0,        Params.initvar      = 1;  end		% nonzero init variance?
if isfield(Params, 'noise')           == 0,        Params.noise        = 100;  end		% noise standard vairance?
if isfield(Params, 'reshuffle')    == 0,        Params.reshuffle  = 1;  end		% SGD over reshuffled data?
if isfield(Params, 'tol')                 == 0,        Params.tol              = 1e-15;  end		% SGD stopping criterion on the loss
if isfield(Params, 'loss')              == 0,        Params.loss           = 1;  end		% compute loss?
if isfield(Params, 'cstep')           == 0,        Params.cstep        = 1;  end		% constant stepsize or not

%% generate linearly separable synthetic data
Xtrain = 2 * (rand(Params.ntrain, Params.d) - .5);
w = 2 * (rand(Params.d, 1) - .5);
ytrain = sign(Xtrain * w);
smin = min(ytrain .* (Xtrain * w));
w = w / smin;
ytrain = sign(Xtrain * w);

Xtest =   1 *randn(Params.ntest, Params.d);
ytest = sign(Xtest * w);

width = 1.2;

for act = 1:length(Params.activations)
    
    act_opt = Params.activations{act};
    
    switch act_opt
        
        case 'nrelu'
            disp('working on noise-injected SGD');
            [f_relu, out_relu] = nReLU(Params, Xtrain, ytrain, Xtest, ytest);
            if Params.loss == 1
                semilogy(f_relu, '-kd', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end
            
        case 'lrelu'
            disp('working on plain-vallina SGD'); % when leaky factor alph=0, then run SGD for ReLUs
            [f_lrelu, out_lrelu] = lReLU(Params, Xtrain, ytrain, Xtest, ytest); 
            if Params.loss == 1
                semilogy(f_lrelu, '-bv', 'LineWidth', width, 'MarkerSize', eps);
                hold on;
            end            

    end
    
end


if Params.loss == 1
    grid
    hx1 = xlabel('Epoch');
    hy1 = ylabel('Training loss');
end

