
clear;

% A sample cost matrix 
M = 5000;
N = 2000;
C = abs(rand(M, N));

% A sample target marginal
targetMarginal = abs(rand(1, N));
targetMarginal = targetMarginal ./sum(targetMarginal);

% Number of prototypes to be selected
m = 200;

% Optional options
options.verbosity = true;
options.useGPU = true;

% Call SPOT
[S, w, gamma, infos] = SPOTgreedy(C, targetMarginal, m, options);