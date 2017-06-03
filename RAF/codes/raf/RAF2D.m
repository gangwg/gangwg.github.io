%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Solving Systems of Random Quadratic Equations via Truncated Amplitude
%  Flow'' by G. Wang, G. B. Giannakis, and Y. C. Eldar.
%  The code below is adapted from implementation of the (Truncated) Wirtinger Flow 
% algorithms implemented by E. Candes, X. Li, M. Soltanolkotabi, and Y. Chen.

function [Relerrs, z] = RAF2D(Psi, x, z, Params, A, At)

M   = numel(Psi);
Relerrs = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro');

for t = 1:Params.T
    
    % TAF updates
    Az = A(z);
    ratio  = abs(Az) ./ Psi; % ratio
    weight = ratio ./ (ratio + Params.eta);
    grad = At(weight .* (Az - Psi .* exp(1i * angle(Az)))) / M;
    
    z = z - Params.muRAF * grad;
    Relerrs = [Relerrs; norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro')]; %#ok<AGROW>

end
