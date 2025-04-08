import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scikeras.wrappers import KerasClassifier # Modern wrapper
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.initializers import RandomUniform, Normal # More specific imports if needed

print("TensorFlow Version:", tf.__version__)

# Data Loading and Initial Inspection
try:
    raw_data = pd.read_csv("wdbc.data", header=None) # Add header=None if needed
    column_names = ['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean',
                    'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                    'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
                    'fractal_dimension_se', 'radius_worst', 'texture_worst',
                    'perimeter_worst', 'area_worst', 'smoothness_worst',
                    'compactness_worst', 'concavity_worst', 'concave points_worst',
                    'symmetry_worst', 'fractal_dimension_worst']
    if raw_data.shape[1] == len(column_names): # Basic check if names fit
         raw_data.columns = column_names
    else:
        print(f"Warning: Loaded data has {raw_data.shape[1]} columns, expected {len(column_names)} based on WDBC names.")
        # If loading worked previously with named columns, your file likely has headers.
        # Re-load without header=None and names if that's the case:
        raw_data = pd.read_csv("wdbc.data")
        print("Re-attempting load assuming headers exist in the file.")

    print("Data Info:")
    raw_data.info()
    print("\nFirst 5 rows:")
    print(raw_data.head())

except FileNotFoundError:
    print("Error: 'wdbc.data' not found. Please ensure the file is in the correct directory.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# Data Preprocessing

# Check if expected columns exist before proceeding
required_feature_cols = ['radius_mean', 'fractal_dimension_worst'] # Check first and last expected feature
target_col = 'diagnosis'

if not all(col in raw_data.columns for col in required_feature_cols) or target_col not in raw_data.columns:
     print("\nError: Required columns ('radius_mean' to 'fractal_dimension_worst', 'diagnosis') not found.")
     print("Available columns:", raw_data.columns.tolist())
     exit()


# Select features (predictors) and target
# Using iloc might be safer if column names are uncertain, but loc is fine if names are correct
# Assuming standard WDBC order where 'diagnosis' is the second column (index 1)
# and features start from the third column (index 2)
# X = raw_data.iloc[:, 2:].values 
# y = raw_data.iloc[:, 1].values  

# Using column names as in the original code
feature_columns = raw_data.loc[:, 'radius_mean':'fractal_dimension_worst'].columns
X = raw_data.loc[:, feature_columns].values
y_raw = raw_data.loc[:, 'diagnosis'].values

# Encode the target variable (Diagnosis: M -> 1, B -> 0)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)
print(f"\nTarget mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")


# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\nShape of scaled features (X_scaled): {X_scaled.shape}")
print(f"Shape of encoded target (y): {y.shape}")

# Neural Network Model Definition

# Function to create the Keras model (required by KerasClassifier)
# Pass input_shape dynamically
def create_model(optimizer='adam', loss='binary_crossentropy', kernel_initializer='glorot_uniform',
                 activation='relu', neurons=16, input_shape=(X_scaled.shape[1],)):
    """Creates a sequential Keras model."""
    model = Sequential(name="WDBC_Classifier")
    model.add(Dense(units=neurons, activation=activation,
                    kernel_initializer=kernel_initializer, input_shape=input_shape))
    model.add(Dropout(0.3)) # Consider if these fixed rates are optimal
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.1))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.3))
    model.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    model.add(Dropout(0.1))
    model.add(Dense(units=1, activation='sigmoid')) # Output layer for binary classification

    # Compile the model inside the function
    model.compile(optimizer=optimizer, loss=loss, metrics=['binary_accuracy'])
    return model

# Hyperparameter Tuning Setup

# Create the SciKeras wrapper
# Pass parameters that are constant or directly controlled by SciKeras here
# Parameters to be tuned by GridSearchCV will be specified in param_grid
model_clf = KerasClassifier(
    model=create_model,  # Pass the function to build the model
    verbose=0,           # Set verbose=0 to prevent Keras output during grid search fitting
    # Pass default values for hyperparameters (GridSearchCV will override these)
    optimizer='adam',
    loss='binary_crossentropy',
    kernel_initializer='glorot_uniform',
    activation='relu',
    neurons=16,
    input_shape=(X_scaled.shape[1],) # Pass input_shape here or ensure create_model handles it
)

# Define the grid of hyperparameters to search
# Assuming scikeras >= 0.6
param_grid = {
    'batch_size': [10, 30],
    'epochs': [50, 100],
    'model__optimizer': ['adam', 'sgd'],
    'model__loss': ['binary_crossentropy', 'hinge'], # Corrected typo 'loos' -> 'loss'
    'model__kernel_initializer': ['random_uniform', 'normal'], # Keras initializers
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [16, 8] # Number of neurons in the hidden layers
}


# Setup GridSearchCV
# n_jobs=-1 uses all available CPU cores for parallelization
grid_search = GridSearchCV(estimator=model_clf,
                           param_grid=param_grid,
                           scoring='accuracy', # Or 'roc_auc', 'f1', etc.
                           cv=5,             # 5-fold cross-validation
                           n_jobs=-1,        # Use parallel processing if possible
                           verbose=1)        # Show progress

# Run Grid Search
print("\nStarting GridSearchCV... (This may take a while)")
grid_search.fit(X_scaled, y) # Use scaled features

# Display Results
print("\nGridSearchCV Complete.")
print(f"Best Parameters found: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_:.4f}")

# You can also access the best estimator directly
best_model = grid_search.best_estimator_
print("\nBest model summary:")
