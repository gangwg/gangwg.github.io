%% Implementation of the Truncated Amplitude Flow algorithm proposed in the paper
%  `` Solving Systems of Random Quadratic Equations via Truncated Amplitude
%  Flow'' by G. Wang, G. B. Giannakis, and Y. C. Eldar.
%  The code below is adapted from implementation of the (Truncated) Wirtinger Flow 
% algorithms implemented by E. Candes, X. Li, M. Soltanolkotabi, and Y. Chen.


clc; close all; clear;

namestr = 'galaxy' ;
stanstr = 'jpg'      ;
X       = mat2gray(imread([namestr,'.',stanstr])) ;
X       = imresize(X, 0.1);
n1      = size(X, 1)                               ;
n2      = size(X, 2)                               ;
alpha_y = 3;

%% TAF/STAF parameters
if exist('Params', 'var')         == 0,  Params.n2           = n2;    end
if isfield(Params, 'n1')          == 0,  Params.n1           = n1; end             % signal dimension

if isfield(Params, 'K')           == 0,  Params.K            = 6;   end
if isfield(Params, 'gamma')       == 0,  Params.gamma        = .7;   end	% thresholding of throwing small entries of Az 0.3 is a good one
if isfield(Params, 'imax')        == 0,  Params.T            = 100;   end	% 
if isfield(Params, 'npower_iter') == 0,  Params.npower_iter  = 100;   end		% number of power iterations
if isfield(Params, 'mu')          == 0,  Params.mu           = 0.6;  end		% step size / learning parameter % originally 0.2

%% noise parameter
SNR = inf;

%% make random masks
Masks = zeros(size(X, 1), size(X, 2), Params.K);
for ll = 1:Params.K, Masks(:, :, ll) = randsrc(size(X, 1), size(X, 2), [1i -1i 1 -1]); end

%% generate data and define linear operators
A = @(I)  fft2(conj(Masks) .* reshape(repmat(I, [1 Params.K]), size(I, 1), size(I, 2), Params.K));
At = @(Y) sum(Masks .* ifft2(Y), 3) * size(Y, 1) * size(Y, 2);
Fn1 = dftmtx(n1);
Fn2 = dftmtx(n2);

parfor rgb = 1:3,
    
    fprintf('Color band %d\n', rgb)
    x       = squeeze(X(:,:,rgb));
    Signal  = abs(A(x));
    noise   = randn(size(Signal));
    Npower  = sum(Signal(:).^2)/sum(noise(:).^2)/10^(SNR/10);
    Psi     = Signal + sqrt(Npower) * noise;
    normest = sqrt(sum((Psi(:)).^2) / numel(Psi));
    

    %% TAF
    z0      = randn(n1, n2);
    z0      = z0 / norm(z0, 'fro');    % Initial guess
    M       = numel(Psi);
    alpha   = 5/6; % orthogonality-promoting initialization parameter % 5/6 by default
    
    ysort   = sort(reshape(Psi, 1, numel(Psi)), 'ascend');
    ythresh = ysort(round(alpha * M));
    ind     = (Psi >= ythresh);
    
    for tt = 1:Params.npower_iter                   %#ok<PFBNS> % Truncated power iterations
        Psi_tr = Psi .* ind;
        z0 = At((A(z0) .* ind));
        z0 = z0 / norm(z0, 'fro');
    end
    
    z = normest * z0;                   % Apply scaling
    IRelerrs_TAF = norm(x - exp(-1i * angle(trace(x' * z))) * z, 'fro') / norm(x, 'fro'); % Initial rel. error
    xinit_opi(:, :, rgb) = z;
    fprintf('OP Initization done!\n');
    
    [Relerrs_TAF(:, rgb), X_TAF(:, :, rgb)] = TAF2D(Psi, x, z, Params, A, At);
    fprintf('TAF done!\n');
  
    
end

fprintf('All done!\n')
figure; imshow(mat2gray(abs(X_TAF)),[]);
figure; imshow(mat2gray(abs(xinit_opi)),[]);
