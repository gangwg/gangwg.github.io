%% Implementation of the noisy-injected stochastic gradient descent proposed in the paper
%  `` Learning ReLU Networks on Linearly Separable Data: Algorithm, Optimality, and Generalization'' 
%   by G. Wang, G. B. Giannakis, and J. Chen, submitted to IEEE Trans. on  Signal Process. on Dec. 2018

function [func, out] = nReLU(Params, Xtrain, ytrain, Xtest, ytest)

orelu = @(x) (x >= 0) .* x;

muu = Params.muu;
func = NaN(Params.T, 1);

W = Params.initvar * randn(Params.k, Params.d);

v = Params.vc * ones(Params.k, 1);
if Params.k >= 2
    v(1:round(Params.k/2)) = - v(1:round(Params.k/2)) ;
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
    
    %% computing loss at every epoch?
    if Params.loss == 1
        if mod(t, 1*Params.ntrain) == 1
            funvalue = sum(orelu(1 - ytrain' .* (v' * orelu(W * Xtrain'))));
            j = j + 1;
            func(j) = funvalue / Params.ntrain;
            if funvalue/Params.ntrain <= Params.tol || funvalue > 1e5
                break;
            end
        end
    end
    
    %% SGD updates
    
    midd = W * Xtrain(it, :)';
    noise = (Params.noise * randn(size(midd))) .* (ytrain(it) * v >= 0);
    
%     if ytrain(it) == -1
%         noise(round(Params.k/2) + 1:end) = 0;
%     else
%         noise(1:round(Params.k/2)) = 0;
%     end
    
    %   W = W + muu * ytrain(it) * max(1 - ytrain(it) * v' * orelu(midd), 0) * (v  * Xtrain(it, :));
    W = W + muu * ytrain(it) * (1 - ytrain(it) * v' * orelu(midd) > eps) * ((v .* ((midd + noise) >= 0)) * Xtrain(it, :));
    %   W = W + muu * (ytrain(it) - v' * orelu(midd)) * ((v .* (max((midd + noise) >= 0, midd >= 0))) * Xtrain(it, :));
    
end

out.error_train = nnz(ytrain ~= (sign(v' * orelu(W * Xtrain'))'))/Params.ntrain;
out.error_test = nnz(ytest ~= (sign(v' * orelu(W * Xtest'))'))/Params.ntest;
out.W = W;

%  disp([out.error_train, out.error_test]);


end