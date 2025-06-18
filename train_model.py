import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Load dataset
data = pd.read_csv("fer2013.csv")
X = []
y = []

for i in range(len(data)):
    pixels = np.array(data['pixels'][i].split(), dtype='float32')
    X.append(pixels.reshape(48, 48, 1))
    y.append(data['emotion'][i])

X = np.array(X) / 255.0
y = to_categorical(np.array(y), num_classes=7)

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Step 5: Save model
model.save("emotion_model.h5")
print(" Model training complete and saved as 'emotion_model.h5'")
