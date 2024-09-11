#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

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
train_df = pd.DataFrame(train1, columns=train.columns)
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

#find mean and std of attribute 1
mean_attribute1 = np.mean(attribute1)
print("Mean of Attribute 1: ", mean_attribute1)

std_attribute1 = np.std(attribute1)
print(f"Standard Deviation of Attribute 1: ", std_attribute1)

#create a span of values for Attribute 1 to range from
x_values = np.linspace(min(attribute1), max(attribute1), 1000)
print(f"X Values: {x_values}")

#calculate the prob density func using normal distribution
pdf_values = norm.pdf(x_values, mean_attribute1, std_attribute1)
print(f"Probality Density Function Values: {pdf_values}")

#plot the probability density function
plt.plot(x_values, pdf_values, label=f'Normal PDF ($\mu$={mean_attribute1:.2f}, $\sigma$={std_attribute1:.2f})')
plt.title("Probablity Density Function fo Attribute 1")
plt.xlabel("Attribute 1")
plt.ylabel("Density")
plt.grid(True)
plt.show()

# choosing Attributes 1 and 2 for the pair plot
pairplot_data = pd.DataFrame(train.iloc[:, [0, 1]], columns=['Attribute 1', 'Attribute 2'])

# create the pairplot using seaborn
sns.pairplot(pairplot_data)
plt.suptitle('Pairplot between Attribute 1 and Attribute 2', y=1.02)  # y adjusts the title position
plt.show()

# variances are on the diagonal of the covariance matrix
variances = np.diag(samp_cov_inner)

# find the largest and smallest variances
max_variance = np.max(variances)
min_variance = np.min(variances)

#find the index of the largest and smallest variances
max_variance_index = np.argmax(variances)
min_variance_index = np.argmin(variances)

print(f"Variance of each attribute: {variances}")
print(f"Largest Variance: Attribute {max_variance_index + 1} with variance {max_variance}")
print(f"Smallest Variance: Attribute {min_variance_index + 1} with variance {min_variance}")

# F\find the largest and smallest covariance (excluding the diagonal)
# set the diagonal to a very small value (or infinity) so it doesn't affect finding min/max covariances
np.fill_diagonal(samp_cov_inner, np.nan)

# find the maximum and minimum covariance values
max_cov = np.nanmax(samp_cov_inner)
min_cov = np.nanmin(samp_cov_inner)

#get the indices of the maximum and minimum covariance values
max_cov_indices = np.unravel_index(np.nanargmax(samp_cov_inner), samp_cov_inner.shape)
min_cov_indices = np.unravel_index(np.nanargmin(samp_cov_inner), samp_cov_inner.shape)

print(f"Covariance matrix: \n{samp_cov_inner}")
print(f"Largest Covariance: Between Attribute {max_cov_indices[0] + 1} and Attribute {max_cov_indices[1] + 1} with covariance {max_cov}")
print(f"Smallest Covariance: Between Attribute {min_cov_indices[0] + 1} and Attribute {min_cov_indices[1] + 1} with covariance {min_cov}")

# assign Attribute 6 and Attribute 8 to variables (index 5 and 7 in 0-based indexing)
attribute_6 = train1[:, 5]  # Attribute 6
attribute_8 = train1[:, 7]  # Attribute 8

# plot Attribute 6 vs Attribute 8 as a scatter plot with circle markers
plt.scatter(attribute_6, attribute_8, marker='o', color='blue')

# add title and labels to the plot
plt.title('Attribute 6 vs Attribute 8')
plt.xlabel('Attribute 6')
plt.ylabel('Attribute 8')
plt.grid(True)
plt.show()

test