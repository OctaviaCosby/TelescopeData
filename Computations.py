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
mean_vector = np.mean(train1)
print(f"Multivatiate Mean Vectors: {mean_vector}")

#create a variable to center the data
centered_func = lambda x: x-x.mean()

centered_data = centered_func(train1)
print(f"Centered Data: {centered_data}")
