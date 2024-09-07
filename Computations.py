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
print(f"Multivatiate Mean Vectors: \n{mean_vector}")

#create a variable to center the data then calculate the centered data
centered_data = train1 - mean_vector
print(f"Centered Data: \n{centered_data}")

#calculate the sample covariance matrix as inner product
n = centered_data.shape[0] #calculate the number of rows
samp_cov_inner = np.dot(centered_data.T, centered_data) /(n - 1)
print(f"Sample Covariance Matrix (Inner Product): \n{samp_cov_inner}")

#calculate the sample covariance matrix as outer product
#initializing the covariance matrix to zeros
samp_cov_outer = np.zeros((centered_data.shape[1], centered_data.shape[1]))

# Calculate the outer product for each centered data point and sum them
for i in range(n):
    outer_product = np.outer(centered_data[i], centered_data[i])
    samp_cov_outer += outer_product

# Normalize by (n-1) to get the sample covariance matrix
samp_cov_outer /= (n - 1)
print(f"Sample Covariance Matrix (Outer Product): \n{samp_cov_outer}")

#create variables for Attribute 1 and Attribute 2
attribute1 = centered_data[:,0]
attribute2 = centered_data[:,1]

#calculate the cosine angle between the vectors
dot_product = np.dot(attribute1, attribute2)
ang_attribute1 = np.linalg.norm(attribute1)
ang_attribute2 = np.linalg.norm(attribute2)
print(f"Attribute 1: {attribute1}")
print(f"Attribute 2: {attribute2}")

#correlation between Attribute 1 and Attribute 2
cos_similarity = dot_product / (ang_attribute1 * ang_attribute2)
print(f"Cosine of the Angle Between Attribute 1 and Attribute 2: {cos_similarity}")

#plot the points
plt.scatter(attribute1, attribute2)
plt.title("Scatter Plot for Corrleation between Attribute 1 & Attribute 2")
plt.xlabel("Attribute 1")
plt.ylabel("Attribute 2")
plt.grid(True)
plt.show()