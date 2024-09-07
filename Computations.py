#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#assign the daata to a new variable
data = pd.read_csv("telescope_data.csv")

#read the data
data.head(201)

#remove the last column because it is a string and we won't be able to compute accurately
train = data.iloc[:,:-1]
print(train)

#convert data to numpy
train1 = train.to_numpy()
print(train1)

#compute the multivariate mean vector
mean_vector = np.mean(train1, axis=0)
print(f"Multivatiate Mean Vectors: {mean_vector}")

#create a variable to center the data then calculate the centered data
centered_data = train1 - mean_vector
print(f"Centered Data: \n{centered_data}")

#calculate the sample covariance matrix as inner product
n = centered_data.shape[0] #calculate the number of rows
samp_cov_inner = np.dot(centered_data.T, centered_data) /(n - 1)
print(f"Sample Covariance Matrix (Inner Product): {samp_cov_inner}")