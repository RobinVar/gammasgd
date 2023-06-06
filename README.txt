####Phase 0: Test correctness by computing OLS on each dataset
#YearPredictionMSD dataset
python3 ols.py "input=../data/YearPredictionMSD_std_train.csv;testset=../data/YearPredictionMSD_std_test.csv"

#cali housing dataset
python3 ols.py "input=../data/cali_train.csv;testset=../data/cali_test.csv"


MINI BATCH SGD W/ GAMMA INITIALIZATION
####Phase 1: produce betas from gamma, save to .csv
#YearPredictionMSD dataset, chunksize used is 1% (5153)
python3 gamma.py "input=../data/YearPredictionMSD_std_train.csv;chunksize=5153;output=YPMSD_beta.csv"

#Cali housing dataset, chunksize used is 1% (144)
python3 gamma.py "input=../data/cali_train.csv;chunksize=144;output=cali_beta.csv"

####Phase 2: Use betas from gamma to initialize mini-batch sgd
#initialization methods
#random
#init=r
#1
#init=1
#0
#init=0
#gamma
#init=g


input = input dataset
testset = dataset used to evaluate
beta = .csv produced from gamma.py of the betas
chunksize= size of each chunk
init = initialization methods (r,1,0,g)
stop = number of steps without improving (early stopping)
trials = number of times to restart model training

#YearPredictionMSD dataset
python3 sksgd.py "input=../data/YearPredictionMSD_std_train.csv;testset=../data/YearPredictionMSD_std_test.csv;beta=YPMSD_beta.csv;chunksize=5153;init=g;stop=10;trials=5"

#cali housing dataset
python3 sksgd.py "input=../data/cali_train.csv;testset=../data/cali_test.csv;beta=cali_beta.csv;chunksize=144;init=g;stop=10;trials=5"