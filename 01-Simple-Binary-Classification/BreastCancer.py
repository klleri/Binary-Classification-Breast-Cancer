import pandas as pd
import numpy as np 

# Scikit-learn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 

from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 

# -----------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------
DATA_FILE = "wdbc.data" # Define filename as a constant
TEST_SPLIT_SIZE = 0.25 # Proportion of data for testing
RANDOM_STATE = 42 # For reproducible train/test splits
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32 # Common batch size default

# Column names based on WDBC dataset description (assuming no header in .data file)
# First column is ID, second is Diagnosis, followed by 30 features.
col_names = ['id', 'diagnosis'] + [f'feature_{i}' for i in range(1, 31)]

# -----------------------------------------------------
# Load and Prepare Data
# -----------------------------------------------------
print(f"Loading data from {DATA_FILE}...")
# Assuming the .data file has no header row
try:
    raw_data = pd.read_csv(DATA_FILE, header=None, names=col_names)
except FileNotFoundError:
    print(f"Error: Data file '{DATA_FILE}' not found. Please ensure it's in the correct directory.")
    exit() # Exit if the file isn't found

print("Data loaded successfully. Basic info:")
raw_data.info()
# print("\nFirst 5 rows:")
# print(raw_data.head()) # Good practice to inspect data

# Separate features (predictors) and target variable
# Drop the 'id' column as it's not a predictive feature
features = raw_data.drop(['id', 'diagnosis'], axis=1).values
target = raw_data['diagnosis'].values
print(f"\nFeatures shape: {features.shape}")
print(f"Target shape: {target.shape}")

# Encode the categorical target variable ('M', 'B') into numerical ( 1, 0)
print("\nEncoding target variable...")
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)
# print(f"Target classes found: {label_encoder.classes_}") # Shows mapping ['B' 'M']
# print(f"First 5 encoded targets: {target_encoded[:5]}")

# -----------------------------------------------------
# Train/Test Split
# -----------------------------------------------------
print(f"\nSplitting data into training and testing sets (Test size: {TEST_SPLIT_SIZE})...")
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target_encoded,
    test_size=TEST_SPLIT_SIZE,
    random_state=RANDOM_STATE, 
    stratify=target_encoded 
)
print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")

# -----------------------------------------------------
# Feature Scaling
# -----------------------------------------------------
# Neural networks generally perform better with scaled features (mean=0, std=1)
print("\nScaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit and transform training data
X_test_scaled = scaler.transform(X_test) # Only transform test data 

# -----------------------------------------------------
# Build Neural Network Model
# -----------------------------------------------------
print("\nBuilding the Neural Network model...")
num_features = X_train_scaled.shape[1] # Get number of features dynamically

model = Sequential(name="BreastCancerClassifier")
# Input layer + first hidden layer
model.add(Dense(units=16, activation='relu', kernel_initializer='he_normal', input_shape=(num_features,)))
# Second hidden layer
model.add(Dense(units=16, activation='relu', kernel_initializer='he_normal'))
# Output layer - 1 unit (binary classification) with sigmoid activation
model.add(Dense(units=1, activation='sigmoid'))

print("Model Summary:")
model.summary()

# -----------------------------------------------------
# Compile Model
# -----------------------------------------------------
print("\nCompiling the model...")
# Define the optimizer (Adam is a good default)
optimizer = Adam(learning_rate=LEARNING_RATE) 

# Compile the model specifying optimizer, loss function and metrics
model.compile(optimizer=optimizer,
              loss='binary_crossentropy', 
              metrics=['binary_accuracy'])

# -----------------------------------------------------
# Train Model
# -----------------------------------------------------
print(f"\nTraining the model for {EPOCHS} epochs with batch size {BATCH_SIZE}...")

# Add EarlyStopping to prevent overfitting
# early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_scaled,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test_scaled, y_test), 
    verbose=1 # Set to 1 or 2 to see progress, 0 for silent
    # callbacks=[early_stopping] # Add callbacks if using
)

print("Model training finished.")

# -----------------------------------------------------
# Evaluate Model
# -----------------------------------------------------
print("\nEvaluating the model on the test set...")

loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Make predictions (probabilities)
print("\nGenerating predictions on the test set...")
y_pred_proba = model.predict(X_test_scaled)

# Convert probabilities to binary class predictions (0 or 1)
y_pred = (y_pred_proba > 0.5).astype(int) # Use 0.5 threshold for sigmoid

# Calculate and display metrics
print("\nCalculating performance metrics...")
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_) # Use original labels

print("\nConfusion Matrix:")
print(conf_matrix)
# [[TN, FP],
#  [FN, TP]]
# TN = True Negatives , FP = False Positives
# FN = False Negatives , TP = True Positives 

print("\nClassification Report:")
print(class_report)

# -----------------------------------------------------
# Optional: Plot training history (requires matplotlib)
# -----------------------------------------------------
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(12, 5))
#
# # Plot Accuracy
# plt.subplot(1, 2, 1)
# plt.plot(history.history['binary_accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_binary_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# # Plot Loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.tight_layout()
# plt.show()
