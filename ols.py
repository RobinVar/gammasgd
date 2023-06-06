import pandas as pd
from sklearn.linear_model import LinearRegression
import sys





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
ts = args.get("testset")

# Load the data from a CSV file
data = pd.read_csv(ds,header=None)

# Separate the predictor variables (X) and target variable (y)
X = data.iloc[:,0:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()


# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Print the coefficients of the model
print("Coefficients:", model.coef_)

# Print the intercept of the model
print("Intercept:", model.intercept_)

# Load the data from a CSV file
data = pd.read_csv(ts,header=None)

# Separate the predictor variables (X) and target variable (y)
X = data.iloc[:,0:-1]
y = data.iloc[:,-1]
r2 = model.score(X, y)
print("r2:",r2 )