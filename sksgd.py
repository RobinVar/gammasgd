from sklearn.linear_model import SGDRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score as R2
from sklearn.metrics import mean_squared_error
import time as tm
import sys







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
ts = args.get("testset")
betas = args.get("beta")
chunk_size= int(args.get("chunksize"))
init_w= args.get("init")
stopsteps= int(args.get("stop"))
trials= int(args.get("trials"))





# ds = "../data/cali_train.csv"
# ts = "../data/cali_test.csv"
#ds ="../data/YearPredictionMSD_std_train.csv" 
#ts ="../data/YearPredictionMSD_std_test.csv"

#cali_beta.csv or YPMDS_beta.csv
beta = pd.read_csv(betas,header=None)
w = np.array(beta.to_numpy().ravel()[1:],copy=True)
b = beta.to_numpy().ravel()[0]

#testset
tf = pd.read_csv(ts,header=None)
ts_y = np.array(tf.iloc[:,-1])
ts_X = np.array(tf.iloc[:,0:-1])

#number of steps with no change before stopping
patience = stopsteps

#chunk_size = 5153 for YPMDS, 144 for cali_housing
#chunk_size = 5153
#chunk_size = 144



avg_r2 = []
avg_chunk = []
for i in range(trials):
    best_r2 = 0
    chunklist = []
    r2list = []
    if (init_w == "r"):
        print("random")
        reg = SGDRegressor()
    else:
        reg = SGDRegressor()

        #Read one row to initialize reg.coef_, otherwise it does not exist
        for df in pd.read_csv(ds,header=None,chunksize=1):
                    X = df.to_numpy()
                    x = X[:,0:-1]
                    y = X[:,-1]
                    reg.partial_fit(x,y)
                    #print(reg.coef_)
                    #reg.coef_ = np.zeros(shape=reg.coef_.shape)
                    #reg.coef_ = w
                    #print(reg.coef_)
                    break  

        if (init_w == "g"):
            w = np.array(beta.to_numpy().ravel()[1:],copy=True)
            reg.coef_ = w
            reg.intercept_= [b]
            print("gamma")
        elif (init_w == "0"):
            reg.coef_ = np.zeros(shape=reg.coef_.shape)
            reg.intercept_= [0]
            print("0")
        elif (init_w == "1"):
            reg.coef_ = np.ones(shape=reg.coef_.shape)
            reg.intercept_= [1]
            print("1")
        print(reg.coef_,reg.intercept_)
    #tm.sleep(2)

    n_iter_no_change = 0
    best_val_error = np.inf
    best_weights = None
    chunk = 1
    r2=0
    for df in pd.read_csv(ds,header=None,chunksize=chunk_size):
        print("chunk: ",chunk)
        chunk+=1

        X = df.to_numpy()
        x = X[:,0:-1]
        y = X[:,-1]
        reg.partial_fit(x,y)

        # Compute the validation error
        y_val_pred = reg.predict(ts_X)
        val_error = mean_squared_error(ts_y, y_val_pred)

        # Check if the validation error has improved
        if val_error < best_val_error:
            best_val_error = val_error
            best_weights = reg.coef_
            n_iter_no_change = 0
            r2 = reg.score(ts_X,ts_y)
            best_r2 = r2
            r2list.append(r2)
        else:
            n_iter_no_change += 1
            # Check if early stopping criterion is met
            if n_iter_no_change >= patience:
                print("Validation error did not improve for {} iterations. Stopping early...".format(patience))
                #print(reg.score(scaled_X,scaled_y))
                chunklist.append(chunk)
                # r2 =reg.score(scaled_X,scaled_y)
                # r2list.append(r2)
                break
        
    del reg
    print(chunk,best_r2)
    avg_chunk.append(chunk)
    avg_r2.append(best_r2)

avg_chunk = sum(avg_chunk) / len(avg_chunk)
print("avg chunks:", avg_chunk)
avg_r2 = sum(avg_r2) / len(avg_r2)
print("avg r2:", avg_r2)
