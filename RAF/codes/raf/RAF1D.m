%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Solving Systems of Random Quadratic Equations via Truncated Amplitude
%  Flow'' by G. Wang, G. B. Giannakis, and Y. C. Eldar.
%  The code below is adapted from implementation of the (Truncated) Wirtinger Flow
% algorithm implemented by E. Candes, X. Li, M. Soltanolkotabi, and Y. Chen.

function [Relerrs, z] = RAF1D(y, x, Params, Amatrix)

%% Initialization
z0      = randn(Params.n1, Params.n2);
z0      = z0 / norm(z0, 'fro');    % Initial guess
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
z = z0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
normest = sqrt(sum(y(:)) / numel(y(:)));    % Estimate norm to scale eigenvector
m       = Params.m;
ymag    = sqrt(y);

[ysort, ~] = sort(y, 'ascend');
ythresh    = ysort(round(m / 1.3));
ind        = y >= ythresh;

Aselect    = Amatrix(ind, :);
weights    = (ymag(ind)).^(Params.alpha); % weights w_i

%% The weighted maximal correlation initialization can be computed using power iterations
%% or the Lanczos algorithm, and the latter performs well when m/n is small
for tt = 1:Params.npower_iter                   % Power iterations
    z  = Aselect' * (weights .* (Aselect * z));
    z  = z / norm(z, 'fro');
end

z = normest * z;
Relerrs = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for t = 1: Params.T
    
    Az    = Amatrix * z; 
    ratio = abs(Az) ./ ymag;
    ang   = Params.cplx_flag * exp(1i * angle(Az)) + (1 - Params.cplx_flag) * sign(Az);
    grad  = Amatrix' * ((ymag .* ang - Az) .* (ratio ./ (ratio + Params.eta))) / m;
    z     = z + Params.muRAF * grad;
    
    Relerrs = [Relerrs; norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro')]; %#ok<AGROW>
    
end


