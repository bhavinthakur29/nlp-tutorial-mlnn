import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Drop missing values if any
df.dropna(inplace=True)

# Map sentiment labels to numeric values (e.g., Positive = 1, Negative = 0)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)

# Pad sequences to ensure equal length
max_length = 100
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Build the model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_length))
model.add(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))) # Bidirectional for better context learning
model.add(Dropout(0.6))
model.add(LSTM(16))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid')) # Sigmoid for binary classification

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])

# Model summary (optional)
model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_padded, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(test_padded, y_test),
    callbacks=[early_stopping]
)

# Use color-blind friendly colors with markers
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy', color='#377eb8', marker='o')  # Blue with circle marker
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='#ff7f00', marker='s')  # Orange with square marker
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Plot training vs validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='#4daf4a', marker='o')  # Green with circle marker
plt.plot(history.history['val_loss'], label='Validation Loss', color='#e41a1c', marker='s')  # Red with square marker
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Predict sentiment on test data
y_pred = (model.predict(test_padded) > 0.5).astype("int32")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Classification Report
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# Sample Prediction to check
new_review = ["The movie was absolutely wonderful!"]
new_seq = tokenizer.texts_to_sequences(new_review)
new_pad = pad_sequences(new_seq, maxlen=max_length, padding='post', truncating='post')
prediction = model.predict(new_pad)[0][0]

if prediction > 0.5:
    print("Positive Review")
else:
    print("Negative Review")