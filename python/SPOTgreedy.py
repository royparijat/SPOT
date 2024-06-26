import cupy as cp
import scipy
from scipy import sparse
import time

def SPOT_GreedySubsetSelection(C, targetMarginal, m):
    # Assumes one source point selected at a time, which simplifies the code.
    # C: Cost matrix of OT: number of source x number of target points {[numY * numX]}
    # targetMarginal: 1 x number of target (row-vector) size histogram of target distribution. Non negative entries summing to 1 {[1*numX]}
    # m: number of prototypes to be selected.

    targetMarginal = targetMarginal / cp.sum(targetMarginal)
    numY = C.shape[0]
    numX = C.shape[1]
    allY = cp.arange(numY)
    # just to make sure we have a row vector.
    targetMarginal = targetMarginal.reshape(1, numX)

    # Intialization
    S = cp.zeros((1, m), dtype=int)
    timeTaken = cp.zeros((1, m), dtype=int)
    setValues = cp.zeros((1, m), dtype=int)
    sizeS = 0
    currOptw = []
    currMinCostValues = cp.ones((1, numX)) * 1000000
    currMinSourceIndex = cp.zeros((1, numX), dtype=int)
    remainingElements = allY
    chosenElements = []
    iterNum = 0
    start = time.time()
    while sizeS < m:
        iterNum = iterNum + 1
        remainingElements = remainingElements[~cp.in1d(
            cp.array(remainingElements), cp.array(chosenElements))]
        temp1 = cp.maximum(currMinCostValues - C, 0)
        temp1 = cp.matmul(temp1, targetMarginal.T)
        incrementValues = temp1[remainingElements]
        maxIncrementIndex = cp.argmax(cp.array(incrementValues))
        # Chosing the best element
        chosenElements = remainingElements[maxIncrementIndex]
        S[0][sizeS] = chosenElements;
        # Updating currMinCostValues and currMinSourceIndex vectors
        tempIndex = (currMinCostValues - C[chosenElements, :]) > 0
        D = C[chosenElements]
        currMinCostValues[tempIndex] = D[tempIndex[0]]
        # currMinSourceIndex reflects index in set S
        currMinSourceIndex[tempIndex] = sizeS
        # Current objective and other booking
        currObjectiveValue = cp.sum(
            cp.dot(
                currMinCostValues,
                targetMarginal.T))
        setValues[0][sizeS] = currObjectiveValue
        if sizeS == m-1 :
            print("targetMarginal", targetMarginal);
            gammaOpt = cp.sparse.csr_matrix((targetMarginal[0], (currMinSourceIndex[0], range(0, numX))), shape=(m, numX));
            print("gammaOpt \n", gammaOpt);
            currOptw = cp.sum(gammaOpt, axis=1).flatten();
            print("currOptw \n", currOptw);
        sizeS = sizeS + 1
    end = time.time()
    print("S : ", S)
    print("Time : ", end - start)
    return S[0]
