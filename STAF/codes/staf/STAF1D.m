%% Implementation of the Stochatsic Truncated Amplitude Flow algorithm proposed in the paper
%  ``Solving Large-scale Systems of Random Quadratic Equations via Stochastic
% Truncated Amplitude Flow'' by G. Wang, G. B. Giannakis, and J. Chen
% The code below is adapted from implementation of the Truncated Wirtinger Flow
% algorithm designed and implemented by E. Candes, X. Li, and M. Soltanolkotabi

function [Relerrs, z] = STAF(y, x, Params, Amatrix)
%% Initialization
z0      = randn(Params.n1, Params.n2); z0 = z0 / norm(z0, 'fro');    % Initial guess
normest = sqrt(sum(y(:)) / numel(y(:)));    % Estimate norm to scale eigenvector

Arnorm  = sqrt(sum(abs(Amatrix).^2, 2)); % norm of rows of Amatrix
ymag    = sqrt(y);
ynorm   = ymag ./ (Arnorm .* normest);

%% finding the angle of inner prodcut

Anorm   = bsxfun(@rdivide, Amatrix, Arnorm);
ysort   = sort(ynorm, 'ascend');
ythresh = ysort(round(Params.m / 1.2));
ind     = (abs(ynorm) >= ythresh);

ms      = sum(ind);
Aaval   = Anorm(ind, :);

%% VR-OPI (variance-reduction OPI)
for tt = 1:Params.npower_iterVR                   % Truncated power iterations
    u  = 1/ms * Aaval' * (Aaval * z0);
    w0 = z0;
    for jj = 1:ms
        %         ii = mod(jj, ms) + 1;
        ii = randi([1, ms], 1);
        w0 = w0 + Params.eta * (Aaval(ii, :)' * (Aaval(ii, :) * (w0 - z0)) + u);
        w0 = w0/norm(w0, 'fro');
    end
    
    z0 = w0;
end

z       = normest * z0;                   % Apply scaling
Relerrs = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error

for t = 1: Params.m * 200  % number of stochastic iterations in terms of data passes
    
    ii  = mod(t, Params.m) + 1;
    %         ii  = randi([1, Params.m], 1);
    Azi = Amatrix(ii, :) * z;
    if abs(Azi) / ymag(ii) <= 1 / (1 + Params.gamma_sgd)% && abs(Azi) / norm(z) >= 1 / (1 + Params.gamma_sgdup)
        continue;
    end
    
    ang = Params.cplx_flag * exp(1i * angle(Azi)) + (1 - Params.cplx_flag) * sign(Azi);
    %     z = z + Params.imuSGD * Amatrix(ii, :)' * (ymag(ii) * ang - Azi) / m;
    z   = z + Amatrix(ii, :)' * (ymag(ii) * ang - Azi) / norm(Amatrix(ii, :))^2;
    
    if mod(t, Params.m) == 1
        Relerrs = [Relerrs; norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro')];
%         if Relerrs(end) <= 1e-16
%             break;
%         end
    end
end
