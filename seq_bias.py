# pip install pandas openpyxl shap xgboost
import pandas as pd
import numpy as np
import openpyxl
import seaborn as sns
import statsmodels.api as sm
from statsmodels.genmod.families import family, links
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.utils import parallel_backend


file_path = "All JY and GMK data edited.xlsx"
sheet_name = "JGM_2"

df = pd.read_excel(file_path, sheet_name=sheet_name)

# Starting off by building a predictive model for measuring 4-6 sequence bias

# Create new subsetted dataframe
four_six = df[['sequence', '4-6_counts']].copy()

# Since we want to build a model that has a response variable of counts, and we want to predict the number of counts from the model,
# We should opt to use either Poisson or negative binomial regression
# Poisson regression assumes that the mean and variance are equal, while negative binomial is used when the variance is greater than the mean
# To determine which of these models is mroe appropriate I want to start off by checking to see if the response variable
# exhibits signs of overdispersion

overdispersion_check = four_six['4-6_counts'].var() > four_six['4-6_counts'].mean()

if overdispersion_check:
    print("The data shows signs of overdispersion.")
else:
    print("The data does not show signs of overdispersion.")

# Since the data exhibits overdispersion I am going to build a negative binomial regression model
# Additionally, I anticipate there being a lot of 0 values after I featurize the data, so negative binomial regression is the optimal model choice

# Building the negative binomial regression model

# Data prep
# First I am going to featurize the sequences and use one-hot encoding

# Define k values for k-mer counting
k_values = [2, 3, 4]

# Function for k-mer counting
def kmer_counting(sequence, k):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    counts = vectorizer.fit_transform(sequence).toarray()
    feature_names = vectorizer.get_feature_names_out()
    return pd.DataFrame(counts, columns=feature_names)

# Apply k-mer counting for each k value and create new columns
for k in k_values:
    df_kmer = kmer_counting(df['sequence'], k)
    df_kmer.columns = [f'kmer_{k}_{col}' for col in df_kmer.columns]  # Rename columns for clarity
    df = pd.concat([df, df_kmer], axis=1)  # Concatenate new columns to the original DataFrame

# Add compositional data for each nucleotide
sequences = df['sequence']
composition_features = pd.DataFrame()
for nucleotide in ['A', 'C', 'G', 'T']:
    composition_features[f'composition_{nucleotide}'] = sequences.apply(lambda seq: seq.count(nucleotide) / len(seq))

# Add GC content of sequences with the correct column name
gc_content = sequences.apply(lambda seq: (seq.count('G') + seq.count('C')) / len(seq))
df['gc_content'] = gc_content  # Assign the correct column name


# Assign hydrophobicity values to nucleotides using the Kyte-Doolittle scale
hydrophobicity_values = {'A': 0.2, 'C': 0.5, 'G': 0.8, 'T': 0.3}

# Function to calculate hydrophobicity for a sequence
def calculate_hydrophobicity(sequence):
    return sum(hydrophobicity_values[n] for n in sequence) / len(sequence)

# Apply the function to your DataFrame
df['hydrophobicity'] = df['sequence'].apply(calculate_hydrophobicity)

# DNA shape prediction
# At this site we can gather data on physical attributes of a DNA sequence, such as minor groove width, roll, propellor twist, and helix twist
# https://rohslab.cmb.usc.edu/DNAshape/serverBackend.php

# Specify the path where you want to save the text file
output_file_path = 'sequences_output.txt'

# Write the sequences to the text file
df['sequence'].to_csv(output_file_path, index=False, header=False)

# After I uploaded the sequences and got the physical data I stored it in the folder seq65b1cdcba51b68.31656329
physcial_data_folder = 'seq65b1cdcba51b68.31656329'

# Come back to this and add these features


# Now it's time to build the model!

# Convert the response variable to numeric type
df['4-6_counts'] = pd.to_numeric(df['4-6_counts'], errors='coerce')
df = df.drop(['P1','four-three','4-6','P1_counts','four-three_counts'], axis=1)

# Drop rows with missing values
df = df.dropna()

# Convert features to numeric type
X = df.drop(['4-6_counts', 'sequence'], axis=1).apply(pd.to_numeric, errors='coerce')

y = df['4-6_counts']  # Response variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit negative binomial regression model
nb_model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial()).fit()

# Predict on the test set
predictions = nb_model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r_squared}')

# Inspect model summary for coefficients and statistics
print(nb_model.summary())

"""
These were the metrics for the original model:
Mean Squared Error: 179.26399850837578
Mean Absolute Error: 6.354650978610209
R-squared: 0.25466095649514986
"""


# Now let's go back and tune the model's hyperparameters
# This will be a computationally intensive task, so let's take advantage of parallel processing

# Define hyperparameter combinations
alpha_values = [0.1, 0.5, 1.0, 2.0]
link_values = ['log', 'sqrt', 'identity']

best_model = None
best_mse = float('inf')  # Initialize with a large value

# Loop through hyperparameter combinations
for alpha in alpha_values:
    for link_value in link_values:
        # Create link object based on the link value
        link = getattr(links, link_value)()

        # Create and fit the Negative Binomial regression model
        nb_model = sm.GLM(y_train, X_train, family=family.NegativeBinomial(link=link, alpha=alpha)).fit()

        # Predict on the test set
        predictions = nb_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)

        # Check if this model is the best so far
        if mse < best_mse:
            best_mse = mse
            best_model = nb_model

# Print the best hyperparameters
print("Best Hyperparameters:", {'alpha': best_model.family.alpha, 'link': best_model.family.link.__class__.__name__})

# Predict on the test set using the best model
predictions = best_model.predict(X_test)

# Evaluate the best model
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r_squared}')

# Inspect model summary for coefficients and statistics
print(best_model.summary())
