%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Solving Systems of Random Quadratic Equations via Truncated Amplitude
%  Flow'' by G. Wang, G. B. Giannakis, and Y. C. Eldar.
%  The code below is adapted from implementation of the (Truncated) Wirtinger Flow 
% algorithm implemented by E. Candes, X. Li, M. Soltanolkotabi, and Y. Chen.

function [Relerrs, z] = TAF1D(y, x, Params, Amatrix)

%% Initialization
z0      = randn(Params.n1, Params.n2); 
z0      = z0 / norm(z0,'fro');    % Initial guess
normest = sqrt(sum(y(:)) / numel(y(:)));    % Estimate norm to scale eigenvector
m       = Params.m;

Arnorm  = sqrt(sum(abs(Amatrix).^2, 2)); % norm of rows of Amatrix
ymag    = sqrt(y);
ynorm   = ymag ./ (Arnorm .* normest);

%% finding the angle of inner prodcut

Anorm      = bsxfun(@rdivide, Amatrix, Arnorm);
[ysort, ~] = sort(ynorm, 'ascend');
ythresh    = ysort(round(Params.m / (6/5))); % 6/5 the orthogonality-promoting initialization parameter
ind        = (abs(ynorm) >= ythresh);

for tt = 1:Params.npower_iter                   % Truncated power iterations
    z0 = Anorm' * (ind .* (Anorm * z0));
    z0 = z0 /norm(z0, 'fro');
end

z          = normest * z0;                   % Apply scaling
Relerrs = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error

for t = 1: Params.T
    
    Az    = Amatrix * z; %A(z);
    ratio = abs(Az) ./ ymag;
    yz    = ratio > 1 / (1 + Params.gamma_lb);
    ang   = Params.cplx_flag * exp(1i * angle(Az)) + (1 - Params.cplx_flag) * sign(Az);
    
    grad  = Amatrix' * (yz .* ymag .* ang - yz .* Az) / m;
    z     = z + Params.mu * grad;
    
    Relerrs = [Relerrs; norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x,'fro')]; %#ok<AGROW>

end

