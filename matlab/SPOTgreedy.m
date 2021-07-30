function [S, currOptw, gammaOpt, infos] = SPOTgreedy(C, targetMarginal, m, options) 
% Assumes one source point selected at a time, which simplifies the code.
% C: Cost matrix of number of source x number of target points
% targetMarginal: 1 x number of target (row-vector) size histogram of target
%                 distribution. Non negative entries summing to 1
% m: number of prototypes to be selected.
% options: optional.
%
% 
% Outputs.
% S: indices of the prototypes selected (row vector).
% currOptw: importance weight of the selected prototypes. Sum(currOptw) = 1.
% gammaOpt: transport plan of the selected prototypes to the target.
% infos: statistics
%
%
% Please cite the below paper (or its published version).
%
% @article{gurumoorthy2021spot,
% title={SPOT: A framework for selection of prototypes using optimal transport},
% author={Gurumoorthy, Karthik S and Jawanpuria, Pratik and Mishra, Bamdev},
% journal={arXiv preprint arXiv:2103.10159},
% year={2021}
% }


    assert(all(targetMarginal>=0));
    targetMarginal = targetMarginal/sum(targetMarginal);

    numY = size(C,1);
    numX = size(C,2);
    allY = 1:numY;

    targetMarginal = reshape(targetMarginal,1,numX); % To make sure we get a row vector.

    % Number of selection per iteration
    k = 1;
    fprintf('Choosing %d elements per iteration in SPOTgreedy\n',k);

    % Options
    if ~isfield(options,'verbosity'); options.verbosity = true; end
    if ~isfield(options,'useGPU'); fprintf('Using GPU \n'); options.useGPU = true; end
    
    printProgress = options.verbosity; % Print progress
    useGPU = options.useGPU; % Use GPU


    % Intialization
    S = zeros(1,m);
    timeTaken = zeros(1,m);
    setValues = zeros(1,m);
    sizeS = 0;
    currOptw = [];
    currMinCostValues = ones(1,numX)*1000000;
    currMinSourceIndex = zeros(1,numX);

    if useGPU
        allY = gpuArray(allY);
        currMinCostValues = gpuArray(currMinCostValues);
        currMinSourceIndex = gpuArray(currMinSourceIndex);
        S = gpuArray(S);
        targetMarginal = gpuArray(targetMarginal);
        C = gpuArray(C);
        setValues = gpuArray(setValues);
        numX = gpuArray(numX);
        numY = gpuArray(numY);
        sizeS = gpuArray(sizeS);
    end

    remainingElements = allY;
    chosenElements = [];
    iterNum = 0;

    while (sizeS < m)
        iterationTime = tic;
        iterNum = iterNum + 1;
        remainingElements = setdiff(remainingElements,chosenElements);
        % incrementValues = zeros(1,length(remainingElements));
        % For each target point, the change in objective is how much 
        % the cost value decreases because of adding this source
        % point i.

        temp1 = (max(currMinCostValues - C,0))*targetMarginal';
        incrementValues = temp1(remainingElements);

        
        [maxIncrementValues, maxIncrementIndex] = max(incrementValues);
        
        % % Uncomment this if statement if we want to have a natural stopping
        % % criterion for selecting the maximum possible number of prototypes.
        % % It exists if there is no further decrease in objective value. 

        % if(maxIncrementValues==0)
        %     fprintf('Adding any more source point does not further decrease the objective.\n');
        %     fprintf('Number of points selected out of required %d elements = %d\n',m,sizeS);
        %     S = S(1:sizeS); % set of selected points
        %     setValues = setValues(1:sizeS); % objective value obtained after adding each point
        %     gammaOpt = sparse(currMinSourceIndex,1:numX,targetMarginal, sizeS, numX);
        %     currOptw = full(sum(gammaOpt,2));
        %     break;
        % end

        % Chosing the best element
        chosenElements = remainingElements(maxIncrementIndex);
        sizeS = sizeS+1;
        S(sizeS) = chosenElements;
        
        % Updating currMinCostValues and currMinSourceIndex vectors
        tempIndex = (currMinCostValues - C(chosenElements,:))>0;
        currMinCostValues(tempIndex) = C(chosenElements,tempIndex);
        currMinSourceIndex(tempIndex) = sizeS; % currMinSourceIndex reflects index in set S
        
        % Current objective and other booking
        currObjectiveValue = sum(currMinCostValues.*targetMarginal);
        setValues(sizeS) = currObjectiveValue;
        timeTaken(iterNum) = toc(iterationTime);
        if(sizeS>=m)
            gammaOpt = sparse(currMinSourceIndex,1:numX,targetMarginal, sizeS, numX);
            currOptw = full(sum(gammaOpt,2));
        end
        if(mod(sizeS, 50) == 0 && printProgress)
            fprintf('Finished choosing %d elements\n', sizeS);
        end

    end
    if printProgress
        fprintf('Time taken to choose the optima set of %d prototypes with SPOTgreedy is = %f secs\n', sizeS, sum(timeTaken));
    end

    % Store statistics
    infos.timeTaken = timeTaken;

end