import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_squared_error

# Set the random seed for reproducibility
np.random.seed(1)

# Define a function to create and compile the CONIP model
def create_CONIP_model(input_shape):
    model = Sequential([
        Conv1D(filters=32, kernel_size=10, activation='relu', padding='same', input_shape=input_shape),
        Dropout(0.1),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=10, activation='relu', padding='same'),
        Dropout(0.05),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.05),
        Dense(1)
    ])
    # Compile model with mean squared error loss function and accuracy metric
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

# Define a function to perform data augmentation
def augment_data(X, num_samples_to_generate, noise_level=0.01):
    augmented_X = []
    for _ in range(num_samples_to_generate):
        sample_index = np.random.choice(len(X))
        sample = X[sample_index]
        
        # Apply random noise
        noise = np.random.normal(0, noise_level, sample.shape)
        augmented_sample = sample + noise
        
        augmented_X.append(augmented_sample)
    
    return np.array(augmented_X)

# Load and preprocess the data
def load_data(file_path):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['velData'])
        Y = np.array(f['MMI_PGA'])
        M = np.array(f['Mag'])
        R = np.array(f['epicentral distance'])
    return X, Y, M, R

# Normalize the input data
def normalize_data(X):
    return X / np.std(X)

# Path to the HDF5 data file 
data_file_path = '/Users/kylechen/OneDrive - The Pennsylvania State University/Senior/S24/DS 340W/ds340w-project/ds340w-chen/Data/NCEDC_new.hdf5'

# Load the data
X, Y, M, R = load_data(data_file_path)

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=42)

# Normalize the input data
train_X_norm = normalize_data(train_X)
test_X_norm = normalize_data(test_X)

# Data augmentation
# After train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=42)

# Calculate a valid num_samples_to_generate that is a multiple of the length of train_X
length_of_train_X = len(train_X)
num_samples_to_generate = length_of_train_X * 5  # for example, to generate 5 times the amount of the original data

# Make sure num_samples_to_generate is a multiple of the length of train_X to avoid fractional repeats
if num_samples_to_generate % length_of_train_X != 0:
    raise ValueError("num_samples_to_generate must be a multiple of the length of train_X")

augmented_train_X = augment_data(train_X, num_samples_to_generate=num_samples_to_generate)  # Number of samples to generate
augmented_train_X_norm = normalize_data(augmented_train_X)

repeats = num_samples_to_generate // len(train_X)
augmented_train_Y = np.repeat(train_Y, repeats = repeats)

assert len(augmented_train_X_norm) == len(augmented_train_Y), "Mismatched data and label sizes."


# Define the model
model = create_CONIP_model(train_X_norm.shape[1:])

# Define callbacks
callbacks_list = [
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10)
]

# Train the model with the original dataset
history_original = model.fit(
    train_X_norm, train_Y, epochs=200, batch_size=64,
    validation_split=0.2, callbacks=callbacks_list, verbose=1
)

# Predict and evaluate the model
pred_test_Y_original = model.predict(test_X_norm)
mse_original = mean_squared_error(test_Y, pred_test_Y_original)
print('Mean squared error on original data:', mse_original)

# Train the model with the augmented dataset
history_augmented = model.fit(
    augmented_train_X_norm, augmented_train_Y, epochs=200, batch_size=64,
    validation_split=0.2, callbacks=callbacks_list, verbose=1
)

# Predict and evaluate the model
pred_test_Y_augmented = model.predict(test_X_norm)
mse_augmented = mean_squared_error(test_Y, pred_test_Y_augmented)
print('Mean squared error on augmented data:', mse_augmented)

# Plot the training and validation loss
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.plot(history_original.history['loss'], label='Training loss (Original)')
plt.plot(history_original.history['val_loss'], label='Validation loss (Original)')
plt.title('Training and Validation Loss (Original Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_augmented.history['loss'], label='Training loss (Augmented)')
plt.plot(history_augmented.history['val_loss'], label='Validation loss (Augmented)')
plt.title('Training and Validation Loss (Augmented Data)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
