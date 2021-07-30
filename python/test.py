from SPOTgreedy import *;
import cupy as cp;
from cupy import random;
import sys;

if len(sys.argv)!=4:
        sys.exit("Please specify 4 arguments along with file name for numY , numX, m. Example Usage -- python test.py 1000 1000 80");

numY = int(sys.argv[1]);
numX = int(sys.argv[2]);
m = int(sys.argv[3]);

if(m>=numY):
        sys.exit("No. of prototypes (m) should be less than size of source (numY)");

C = random.randint(100, size=(numY, numX));
t = random.randint(100, size = numX);

results = SPOT_GreedySubsetSelection(C,t,m);
print("Prototype Indices -- " , results);
