import pandas as pd 
import numpy as np 
import sys
import time as tm
from sklearn.metrics import r2_score as R2
import sys

def compute_gamma(X):
    ones = [1]*X.shape[1]
    Z = np.vstack([ones,X])
    gamma = np.dot(Z,np.transpose(Z))
    return gamma

def gamma_np(ds):
    iteration= 0
    max_iter = 1
    for df in pd.read_csv(ds,chunksize=chunksize):
        if iteration == max_iter:break
        X = df.to_numpy()

        #print("iterations: ",iteration)
        if iteration == 0:
            gamma = compute_gamma(np.transpose(X))
        else:
            partial_gamma = compute_gamma(np.transpose(X))
            gamma += partial_gamma
        #print(gamma)
        iteration +=1
        
    #print(gamma)
    return gamma


def LR_gamma(gamma):
    #print(gamma.shape)
    d = len(gamma)
    Q = gamma[0:d-1,0:d-1] #[row,column]
    XYT = gamma[0:d-1,d-1]
    invQ = np.linalg.pinv(Q)
    beta = np.matmul(invQ,XYT)
    return beta


#######MAIN
# Get the argument string
arg_string = sys.argv[1]  # Assuming the argument is passed as the first command-line argument

# Split the argument string into key-value pairs
pairs = arg_string.split(";")

# Create a dictionary to store the parsed arguments
args = {}

# Iterate over the key-value pairs and extract the arguments
for pair in pairs:
    key, value = pair.split("=")
    args[key] = value

# Access the parsed arguments
ds = args.get("input")
chunksize= int(args.get("chunksize"))
output_fn = args.get("output")

# Use the parsed arguments as needed
# print("arg1:", arg1_value)
# print("arg2:", arg2_value)
# print("arg3:", arg3_value)

#Datasets used
#ds = "../data/cali_train.csv"
#ts = "../data/cali_test.csv"
#ds ="../data/YearPredictionMSD_std_train.csv" 
#ts ="../data/YearPredictionMSD_std_test.csv"

#chunksize = 5153 for YPMDS, 144 for cali_housing
#chunksize = 5153
#chunksize = 144


#start = tm.time()
beta = LR_gamma(gamma_np(ds))
#tot = tm.time() - start
#print("tot: ",tot)
#print(beta)

####CHECK R2 with testset
#df = pd.read_csv(ts)
#df = pd.read_csv(ds,header=None)
#scaled_y = np.array(df.iloc[:,-1])
#scaled_X = np.array(df.iloc[:,0:-1])
#preds = np.dot(scaled_X,beta[1:]) + beta[0]
#r2 = R2(scaled_y,preds)
#print(beta)


#YPMDS_beta.csv or cali_beta.csv
beta.tofile(output_fn,sep=',')
#beta.tofile('cali_beta.csv',sep=',')
