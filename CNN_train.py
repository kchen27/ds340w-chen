import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import r2_score
from scipy.stats import norm
# Try subset of the data
# Work on the augmentation
# Consider about the actual methods
# Have the results and show the problems that either exists or not

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
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def augment_data(X, num_stations_subset, blinding_time_range):
    """
    Perform data augmentation by selecting a subset of stations and applying temporal blinding.

    Parameters:
    - X: The original dataset (num_samples, num_stations, num_features)
    - num_stations_subset: Number of stations to include in the subset
    - blinding_time_range: Tuple indicating the range (min, max) from which to randomly choose the blinding time

    Returns:
    - Augmented data
    """
    augmented_X = []
    num_samples, num_stations, num_features = X.shape
    
    for sample in X:
        # Randomly select a subset of stations
        stations_indices = np.random.choice(num_stations, num_stations_subset, replace=False)
        sample_subset = sample[stations_indices, :]
        
        # Apply temporal blinding by zeroing waveforms after a random time t_1
        t_1 = np.random.uniform(blinding_time_range[0], blinding_time_range[1])
        time_index = int(t_1 * num_features)  # Assuming uniform time distribution in features
        sample_blinded = np.array([np.where(np.arange(num_features) > time_index, 0, station) for station in sample_subset])
        
        augmented_X.append(sample_blinded)
    
    return np.array(augmented_X)


# Load the data
path = '/Users/kylechen/OneDrive - The Pennsylvania State University/Senior/S24/DS 340W/ds340w-project/ds340w-chen'
fileName = '/Users/kylechen/OneDrive - The Pennsylvania State University/Senior/S24/DS 340W/ds340w-project/ds340w-chen/Data/NCEDC_new.hdf5'
with h5py.File(os.path.join(path, fileName), 'r') as f:
    X = np.array(f['velData'])
    Y = np.array(f['MMI_PGA'])
    M = np.array(f['Mag'])
    R = np.array(f['epicentral distance'])

# Split the data into training and testing sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.4, random_state=42)

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

def datasetPlot(x, y, xlabel, ylabel, xbin, ybin, ymin=-3, ymax=8, figSavePath=False, figName=False):
    # data check
    plt.figure()
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    
    ax1.hist(x, bins=list(np .arange(min(x), max(x)+xbin, xbin)), 
             facecolor="blue", edgecolor="black", alpha=0.7)
    ax2.plot(x, y, 'bx', label='N='+str(len(x)))
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_ylim(ymin, ymax)
    ax2.legend()
    ax3.hist(y, facecolor="cyan", edgecolor="black", alpha=0.7, 
             bins=list(np.arange(ymin, ymax+ybin, ybin)), orientation='horizontal')
    ax3.set_ylim(ymin, ymax)
    plt.tight_layout()
    
    if figSavePath and figName:
        plt.savefig(figSavePath+figName, dpi=200) #, transparent=True
    plt.show()

