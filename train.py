from preprocess import load_data, LANGUAGES
from cnn_model import build_cnn_model
from tensorflow.keras.callbacks import EarlyStopping

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
loss, acc = model.evaluate(X_test, y_test)
print(f"CNN Accuracy: {acc:.3f}")

# 5. Save model
model.save("cnn_language_model.h5")
