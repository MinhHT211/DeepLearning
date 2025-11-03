from preprocess import load_data, LANGUAGES
from cnn_model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# 1. Load data
X_train, X_test, y_train, y_test = load_data("data")

# 2. Build CNN
model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=len(LANGUAGES))

# 3. Training
es = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    callbacks=[es]
)

# 4. Evaluate 
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# switch to index layer

y_true = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test

# Prediction
y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Metrics 
acc = accuracy_score(y_true, y_pred)
prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
prec_w = precision_score(y_true, y_pred, average='weighted', zero_division=0)
recall_w = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\n=== TEST METRICS ===")
print(f"Accuracy        : {acc:.4f}")
print(f"Precision Macro : {prec_macro:.4f}")
print(f"Recall Macro    : {recall_macro:.4f}")
print(f"F1-score Macro  : {f1_macro:.4f}")
print(f"Precision Weighted : {prec_w:.4f}")
print(f"Recall Weighted    : {recall_w:.4f}")
print(f"F1-score Weighted  : {f1_w:.4f}")

# Report by language
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_true, y_pred, target_names=LANGUAGES, zero_division=0))

# Confusion Matrix 
cm = confusion_matrix(y_true, y_pred)
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111)
im = ax.imshow(cm)  
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')

# Tick ​​label if the number of classes is not too large
ax.set_xticks(range(len(LANGUAGES)))
ax.set_yticks(range(len(LANGUAGES)))
ax.set_xticklabels(LANGUAGES, rotation=90)
ax.set_yticklabels(LANGUAGES)

# Print numbers on each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha='center', va='center')

plt.tight_layout()
plt.show()


# 5. Save model
model.save("cnn_language_model.h5")

# --- PLOTTING THE CURVES ---

print("Plotting training and validation curves...")

# Keras có khi lưu key là 'accuracy' hoặc 'categorical_accuracy'
acc_key = 'accuracy' if 'accuracy' in history.history else 'categorical_accuracy'
val_acc_key = 'val_accuracy' if 'val_accuracy' in history.history else 'val_categorical_accuracy'

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history[acc_key], label='Training Accuracy', color='green')
plt.plot(history.history[val_acc_key], label='Validation Accuracy',color='orange')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss',color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss',color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
