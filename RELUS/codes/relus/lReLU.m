%% Implementation of the noisy-injected stochastic gradient descent proposed in the paper
%  `` Learning ReLU Networks on Linearly Separable Data: Algorithm, Optimality, and Generalization'' by G. Wang, G. B. Giannakis, and J. Chen.

%% This is plain-vanilla SGD with alph=0
function [func, out] = lReLU(Params, Xtrain, ytrain, Xtest, ytest)

alph = 0;
orelu = @(x) (x > 0) .* x;
olrelu = @(x) (x > 0) .* x + alph * ((x <= 0) .* x);

muu = Params.muu;
func = NaN(Params.T, 1);

W = Params.initvar * randn(Params.k, Params.d);

v = Params.vc * ones(Params.k, 1);
if Params.k >= 2
    v(1:round(Params.k/2)) = -Params.vc;
end
j = 0;

for t = 1: Params.ntrain*Params.T
    
    if Params.reshuffle == 1
        if mod(t, Params.ntrain) == 1
            it = 0;
            index = randperm(Params.ntrain, Params.ntrain);
        end
        it = it + 1;
    else
        it = randi(Params.ntrain, 1);
        xused(it) = 1;
    end
    
    %% diminishing stepsize
    if Params.cstep == 0
        muu = Params.muu / mod(t, Params.ntrain);
    end
    
    %% computing loss?
    if Params.loss == 1
        if mod(t, Params.ntrain) == 1
            funvalue = sum(orelu(1 - ytrain' .* (v' * olrelu(W * Xtrain'))));
            j = j + 1;
            func(j) = funvalue / Params.ntrain + eps;
            if funvalue/Params.ntrain <= Params.tol || funvalue > 1e5
                break;
            end
        end
    end
    
    %% SGD updates
    
    midd = W * Xtrain(it, :)';
    W = W + muu * ytrain(it) * (1 - ytrain(it) * v' *olrelu(midd) > eps) * ((v .* ((midd > 0) + alph * (midd<= 0))) * Xtrain(it, :));
    
end

out.error_train = nnz(ytrain ~= (sign(v' * olrelu(W * Xtrain'))'))/Params.ntrain;
out.W = W;

if Params.error == 1
    out.error_test = nnz(ytest ~= (sign(v' * orelu(W * Xtest'))'))/Params.ntest;
    disp([out.error_train, out.error_test]);
end

% 1 - ytrain' .* (v' * olrelu(W * Xtrain'))
% orelu(1 - ytrain' .* (v' * olrelu(W * Xtrain')))

end
