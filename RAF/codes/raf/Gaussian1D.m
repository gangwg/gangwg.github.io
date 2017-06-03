%% Implementation of the Reweighted Amplitude Flow algorithm proposed in the paper
%  `` Solving Almost Systems of Random Quadratic Equations’’
%  by G. Wang, G. B. Giannakis, Y. Saad, and J. Chen.
%  The code below is adapted from implementation of the Wirtinger Flow
% algorithm implemented by E. Candes, X. Li, and M. Soltanolkotabi.

clear;
clc;
close all;

if exist('Params', 'var')           == 0,  Params.n2            = 1;    end
if isfield(Params, 'n1')            == 0,  Params.n1            = 1000; end             % signal dimension
if isfield(Params, 'm')             == 0,  Params.m             = floor(2 * Params.n1) - 1;  end     % number of measurements
if isfield(Params, 'cplx_flag')     == 0,  Params.cplx_flag     = 0;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'T')             == 0,  Params.T             = 2000;  end    	% number of gradient iterations
if isfield(Params, 'npower_iter')   == 0,  Params.npower_iter   = 200;   end		% number of power iterations
if isfield(Params, 'eta')           == 0,  Params.eta           = 10;   end	% weighting parameter in the gradient flow
if isfield(Params, 'alpha')         == 0,  Params.alpha         = 0.5;   end	% weighting parameter in the initialization
if isfield(Params, 'muRAF')         == 0,  Params.muRAF         = 2 * (1 - Params.cplx_flag) + 5 * Params.cplx_flag;  end

display(Params);

%% Make signal and data (noiseless)
Amatrix = (1 * randn(Params.m, Params.n1) + Params.cplx_flag * 1i * randn(Params.m, Params.n1)) / (sqrt(2)^Params.cplx_flag);
x       = 1 * randn(Params.n1, 1) + Params.cplx_flag * 1i * randn(Params.n1, 1);

A       = @(I) Amatrix  * I;
At      = @(Y) Amatrix' * Y;

var     = 0e-1;
noise   = var * randn(Params.m, 1);
y       = (abs(A(x)) + noise).^2;


%% run RAF algorithm
[Relerrs_RAF, z] = RAF1D(y, x, Params, Amatrix); 
disp('----------RAF done!----------');

%% plot the relative error of TAF
figure,
semilogy(Relerrs_RAF)
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
    title('RAF: Relerr vs. itercount')
grid


