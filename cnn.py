from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# Initialize the classifier
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolution layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compile the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare the image data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('27CNN_Cinsiyet/veriler/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('27CNN_Cinsiyet/veriler/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# Train the classifier
classifier.fit(training_set,
               steps_per_epoch=8000 // 32,
               epochs=25,
               validation_data=test_set,
               validation_steps=2000 // 32)

# Reset test set generator for prediction
test_set.reset()
pred = classifier.predict(test_set, verbose=1)
pred = (pred > 0.5).astype(int)

print('Prediction complete')

# Get the true labels
test_labels = []
for i in range(int(np.ceil(test_set.samples / test_set.batch_size))):
    test_labels.extend(np.array(test_set[i][1]))

print('Test labels extracted')

# Get the file names
file_names = test_set.filenames

# Create a DataFrame with results
results = pd.DataFrame({
    'file_names': file_names,
    'predictions': pred.flatten(),
    'true_labels': test_labels
})

# Calculate confusion matrix
cm = confusion_matrix(test_labels, pred)
print('Confusion Matrix:')
print(cm)
