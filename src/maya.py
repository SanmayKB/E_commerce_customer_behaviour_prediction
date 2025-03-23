import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


file_path = "data/ecommerce_customer_data_custom_ratios.csv"  
df = pd.read_csv(file_path)


features = ["Product Price", "Quantity", "Total Purchase Amount", "Payment Method", "Returns"]
target = "Product Category"


df = df.dropna(subset=[target] + features)


le_payment = LabelEncoder()
df["Payment Method"] = le_payment.fit_transform(df["Payment Method"])


target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])


scaler = StandardScaler()
df[["Product Price", "Quantity", "Total Purchase Amount"]] = scaler.fit_transform(
    df[["Product Price", "Quantity", "Total Purchase Amount"]]
)


X = df[features].values
y = df[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


num_classes = len(np.unique(y))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(num_classes, activation="softmax")  # Output layer
])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50, batch_size=32, verbose=1
)


y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

accuracy = accuracy_score(y_test_labels, y_pred)
report = classification_report(y_test_labels, y_pred, target_names=target_encoder.classes_)

print(f"Neural Network Model Accuracy: {accuracy:.4f}")
print("Detailed Classification Report:")
print(report)
