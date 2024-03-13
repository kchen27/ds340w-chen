import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
#Authors Alvin Chen and Kyle Chen
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import r2_score
from scipy.stats import norm

# Set the random seed for reproducibility
np.random.seed(1)

# Define a function to calculate MMI using the method by Worden et al. 2016
def calculate_MMI(PGM, R, M, case):
    if case == 'PGA':
        coefficients = [1.78, 1.55, -1.60, 3.70, -0.91, 1.02, -0.17, 4.22]
    elif case == 'PGV':
        coefficients = [3.78, 1.47, 2.89, 3.16, 0.90, 0.00, -0.18, 4.56]
    else:
        raise ValueError("Case must be 'PGA' or 'PGV'")
    
    c1, c2, c3, c4, c5, c6, c7, t = coefficients
    MMI = c1 + c2 * np.log10(PGM) + c5 + c6 * np.log10(R) + c7 * M
    if MMI > t:
        MMI = c3 + c4 * np.log10(PGM) + c5 + c6 * np.log10(R) + c7 * M
    return MMI

# Define a function to create and compile the CNN model
def create_CNN_model(input_shape, optimizer='adam', loss='mse'):
    model = Sequential([
        Conv1D(filters=32, kernel_size=20, activation='relu', input_shape=input_shape),
        Dropout(0.1),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=20, activation='relu'),
        Dropout(0.05),
        Flatten(),
        Dense(100, activation='relu'),
        Dropout(0.05),
        Dense(1)
    ])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# Load the data
path = '/Users/kylechen/Library/CloudStorage/OneDrive-ThePennsylvaniaStateUniversity/Senior/S24/DS 340W/ds340w-project/ds340w-chen'
fileName = 'NCEDC_new.hdf5'
with h5py.File(os.path.join(path, fileName), 'r') as f:
    X = np.array(f['velData'])
    Y = np.array(f['MMI_PGA'])
    M = np.array(f['Mag'])
    R = np.array(f['epicentral distance'])

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=42)

# Normalize the input data
train_X = train_X / np.std(train_X)
test_X = test_X / np.std(test_X)

# Define the model
model = create_CNN_model(train_X.shape[1:])

# Define callbacks
callbacks_list = [
    ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10)
]

# Train the model
history = model.fit(train_X, train_Y, epochs=200, batch_size=64, validation_split=0.2, callbacks=callbacks_list, verbose=1)

# Predict and evaluate the model
pred_test_Y = model.predict(test_X)
test_Y = np.nan_to_num(test_Y)
pred_acc = r2_score(test_Y, pred_test_Y)
print('Prediction accuracy:', pred_acc)

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
