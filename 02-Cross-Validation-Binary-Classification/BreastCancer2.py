import pandas as pd
import numpy as np
import time 

# Scikit-learn modules
from sklearn.model_selection import cross_val_score, StratifiedKFold # StratifiedKFold is often better for classification CV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline 

# Keras / TensorFlow modules
# KerasClassifier is now part of SciKeras (pip install scikeras)
# If using older TensorFlow/Keras, the original import might work, but SciKeras is the modern way.
# Option 1: Modern approach
try:
    from scikeras.wrappers import KerasClassifier
except ImportError:
    print("Warning: SciKeras not found. Falling back to tf.keras.wrappers.scikit_learn.")
    print("Consider installing scikeras: pip install scikeras")
    # Option 2: Older fallback 
    try:
        from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
    except ImportError:
        print("Error: Cannot find KerasClassifier. Ensure TensorFlow/Keras or SciKeras is installed.")
        exit()

from tensorflow import keras # Modern way to import Keras components
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------
# Configuration & Constants
# ------------------------------
DATA_FILE = "wdbc.data"
RANDOM_STATE = 42 # For reproducibility if needed anywhere 
N_SPLITS = 10 # Number of folds for cross-validation
LEARNING_RATE = 0.001
EPOCHS = 100 # Will run for 100 epochs for EACH CV fold
BATCH_SIZE = 32 # Larger batch size than 10 is usually better/faster

# Column names for WDBC dataset (assuming no header in .data file)
col_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# ------------------------------
# Load and Prepare Data
# ------------------------------
print(f"Loading data from {DATA_FILE}...")
try:
    raw_data = pd.read_csv(DATA_FILE, header=None, names=col_names)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found.")
    exit()

print("Data loaded successfully.")
# raw_data.info() # Debugging

# Separate features (predictors) and target
features = raw_data.drop(['id', 'diagnosis'], axis=1).values
target_labels = raw_data['diagnosis'].values
print(f"\nFeatures shape: {features.shape}")
print(f"Target shape: {target_labels.shape}")

# Encode target variable
print("\nEncoding target variable...")
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target_labels)
num_features = features.shape[1] # Get number of features for model input

# -------------------------------------
# Define Keras Model Building Function
# ------------------------------------

def create_model(input_dims, learning_rate=0.001):
    """Builds and compiles the Keras Sequential model."""
    model = Sequential(name="BreastCancerClassifier_CV")
    # Using 'he_normal' initializer (good for ReLU) and dynamic input_shape
    model.add(Dense(units=16, activation='relu', kernel_initializer='he_normal', input_shape=(input_dims,)))
    model.add(Dropout(0.2)) # Regularization
    model.add(Dense(units=16, activation='relu', kernel_initializer='he_normal'))
    model.add(Dropout(0.2)) # Regularization
    model.add(Dense(units=1, activation='sigmoid')) # Output for binary classification

    optimizer = Adam(learning_rate=learning_rate) # Use configured learning rate

    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    return model

# -----------------------------------------------------
# Create Scikit-learn Pipeline with Scaler and Keras Model
# -----------------------------------------------------
print("\nCreating Scikit-learn pipeline with StandardScaler and KerasClassifier...")

# Instantiate the KerasClassifier wrapper
# Pass hyperparameters for training (epochs, batch_size) and model creation (input_dims, learning_rate)
# Use verbose = 0 to keep CV output clean
keras_estimator = KerasClassifier(
    model=create_model, # Pass the function reference
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    input_dims=num_features,
    learning_rate=LEARNING_RATE
)

# Create the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Scale data
    ('keras_model', keras_estimator) # Keras model
])

# ---------------------------
# Perform Cross-Validation
# ---------------------------
print(f"\nPerforming {N_SPLITS}-fold cross-validation (this may take some time)...")

# Use StratifiedKFold for classification tasks to maintain class balance in folds
cv_strategy = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

start_time = time.time()
# Perform cross-validation on the entire pipeline
# This ensures scaling happens independently within each fold using only that fold's training data
results = cross_val_score(
    estimator=pipeline, 
    X=features,
    y=target_encoded,
    cv=cv_strategy, 
    scoring='accuracy',
    n_jobs=1 
)
end_time = time.time()

print(f"Cross-validation finished in {end_time - start_time:.2f} seconds.")

# ---------------------
# Analyze Results
# ---------------------
mean_accuracy = results.mean()
std_dev_accuracy = results.std()

print("\n--- Cross-Validation Results ---")
print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in results]}")
print(f"Mean Accuracy: {mean_accuracy:.4f} ({mean_accuracy*100:.2f}%)")
print(f"Standard Deviation of Accuracy: {std_dev_accuracy:.4f}")
print("---------------------------------")


if std_dev_accuracy > 0.05: # Example threshold, adjust as needed
    print("\nThe standard deviation is relatively high, suggesting the model's performance")
    print(" Varies significantly across different data subsets (potential overfitting or instability).")
else:
     print("\nNote: The standard deviation is relatively low, indicating consistent performance across folds.")
