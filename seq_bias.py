# piip install pandas openpyxl shap xgboost
import pandas as pd
import openpyxl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import GridSearchCV

file_path = "All JY and GMK data edited.xlsx"
sheet_name = "JGM_2"

df = pd.read_excel(file_path, sheet_name=sheet_name)

# fasta file creation
fasta_filename = 'sequences.fasta'

with open(fasta_filename, 'w') as fasta_file:
    for index, row in df.iterrows():
        sequence_id = f">{index}"
        sequence_data = row['sequence']
        fasta_file.write(f"{sequence_id}\n{sequence_data}\n")

print(f"FASTA file '{fasta_filename}' created successfully.")

# make a heat map here


#

# predictive model for 4-6 sequence bias

four_six = df[['sequence', '4-6_counts']].copy()

overdispersion_check = four_six['4-6_counts'].var() > four_six['4-6_counts'].mean()

if overdispersion_check:
    print("The data shows signs of overdispersion.")
else:
    print("The data does not show signs of overdispersion.")

# Calculate hairpin formation

# One-hot encode the 'sequence' column
one_hot_encoded_four_six = pd.get_dummies(four_six['sequence'].apply(lambda x: ' '.join(list(x))))

# Define k values for k-mer counting
k_values = [2, 3, 4]

# Concatenate the one-hot encoded features with the original DataFrame
df_encoded = pd.concat([four_six, one_hot_encoded_four_six], axis=1)

# Function for k-mer counting
def kmer_counting(sequence, k):
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
    counts = vectorizer.fit_transform([sequence]).toarray()
    feature_names = vectorizer.get_feature_names_out()
    return pd.Series(counts.flatten(), index=feature_names)

# Apply k-mer counting for each k value and create new columns
for k in k_values:
    df_kmer = df['sequence'].apply(lambda seq: kmer_counting(seq, k))
    df_kmer = df_kmer.notna().astype(int)  # Convert non-NaN values to 1, NaN to 0
    df_encoded = pd.concat([df_encoded, df_kmer], axis=1)

# Display the resulting DataFrame
print(df_encoded)

# Predictive Modeling
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


df_features = df_encoded.drop('sequence', axis=1)

# Handle NaN values in the target variable '4-6_counts' using mean imputation
imputer = SimpleImputer(strategy='mean')
df_features['4-6_counts'] = imputer.fit_transform(df_features[['4-6_counts']])

# Normalize the target variable
scaler = StandardScaler()
df_features['4-6_counts'] = scaler.fit_transform(df_features[['4-6_counts']])

# Define features and target
X = df_features.drop('4-6_counts', axis=1)
y = df_features['4-6_counts']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_mse = mean_squared_error(y_test, linear_predictions)
linear_r2 = r2_score(y_test, linear_predictions)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

# Gradient Boosting

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4]
}

# Create the GradientBoostingRegressor
gb_model = GradientBoostingRegressor(random_state=42)

# Perform GridSearchCV
grid_search = GridSearchCV(gb_model, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train a model with the best parameters
best_gb_model = GradientBoostingRegressor(**best_params, random_state=42)
best_gb_model.fit(X_train, y_train)

# Make predictions and evaluate the model
best_gb_predictions = best_gb_model.predict(X_test)
best_gb_mse = mean_squared_error(y_test, best_gb_predictions)
best_gb_r2 = r2_score(y_test, best_gb_predictions)

# Neural Net
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
# Compile the model with adjusted learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Train the model with adjusted batch size and epochs
model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model on the test set
nn_predictions = model.predict(X_test).flatten()
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)

# Display results
print(f'Linear Regression - Mean Squared Error: {linear_mse}, R-squared: {linear_r2}')
print(f'Random Forest - Mean Squared Error: {rf_mse}, R-squared: {rf_r2}')
print(f'Gradient Boosting - Mean Squared Error: {best_gb_mse}, R-squared: {best_gb_r2}')
print(f'Neural Network - Mean Squared Error: {nn_mse}, R-squared: {nn_r2}')


# Testing the gradient boosting model on the other polymerases

best_gb_predictions = best_gb_model.predict(X_test)
best_gb_mse = mean_squared_error(y_test, best_gb_predictions)
best_gb_r2 = r2_score(y_test, best_gb_predictions)


# Since the best model was the gradient boosting model I want to see the feature importance of that model

feature_importance = best_gb_model.feature_importances_

import shap
import xgboost

# Convert scikit-learn model to XGBoost model
xgb_model = xgboost.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Create a tree explainer
explainer = shap.TreeExplainer(xgb_model)

# Get SHAP values
shap_values = explainer.shap_values(X_test)

# Plot summary plot
shap.summary_plot(shap_values, X_test)



#check for overdispersion
import statsmodels.api as sm

# Assuming 'response' is the count data column
model_poisson = sm.GLM(df_encoded['response'], df_encoded.iloc[:, 2:], family=sm.families.Poisson())
result_poisson = model_poisson.fit()
print(result_poisson.summary())

residuals = result_poisson.resid_pearson
plt.scatter(result_poisson.fittedvalues, residuals)

model_negbinom = sm.GLM(df_encoded['response'], df_encoded.iloc[:, 2:], family=sm.families.NegativeBinomial())
result_negbinom = model_negbinom.fit()
print(result_negbinom.summary())

print("Poisson AIC:", result_poisson.aic)
print("Negative Binomial AIC:", result_negbinom.aic)



from sklearn.model_selection import train_test_split

# Assuming 'response' is the count data column
X_train, X_test, y_train, y_test = train_test_split(df_encoded.iloc[:, 2:], df_encoded['response'], test_size=0.2, random_state=42)

model_negbinom = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
result_negbinom = model_negbinom.fit()

predictions = result_negbinom.predict(X_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
