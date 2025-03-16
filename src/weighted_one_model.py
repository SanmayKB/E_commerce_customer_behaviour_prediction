import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures

# Load dataset
file_path = "data/ecommerce_customer_data_custom_ratios.csv"
df = pd.read_csv(file_path)

# Selecting only the required columns
selected_features = ["Product Price", "Quantity", "Total Purchase Amount"]
target_column = "Product Category"

# Handle missing values
df = df.dropna(subset=[target_column] + selected_features)

# Feature Engineering: Adding More Non-Linearity
df["Price_per_Unit"] = df["Total Purchase Amount"] / (df["Quantity"] + 1)  
df["Purchase_Sqrt"] = np.sqrt(df["Total Purchase Amount"])
df["Price_Quantity_Interaction"] = df["Product Price"] * df["Quantity"]

# Polynomial Features (Degree 2)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[selected_features])
poly_feature_names = poly.get_feature_names_out(selected_features)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Concatenate original and polynomial features
df = pd.concat([df, df_poly], axis=1)

# Final selected features
selected_features += ["Price_per_Unit", "Purchase_Sqrt", "Price_Quantity_Interaction"]
selected_features.extend(poly_feature_names)

# Encode the target variable
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Splitting data into features and target
X = df[selected_features]
y = df[target_column]

# Feature Scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert data into LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define Optimized LightGBM Parameters
params = {
    "boosting_type": "gbdt",
    "objective": "multiclass",
    "num_class": len(df[target_column].unique()),
    "learning_rate": 0.1,  
    "num_leaves": 120,  
    "max_depth": 25,  
    "min_child_samples": 5,  
    "subsample": 0.95,  
    "colsample_bytree": 0.95,  
    "reg_lambda": 1.2,
    "reg_alpha": 1.2,
    "class_weight": "balanced",  
    "metric": "multi_logloss",
}

# Train the model
model = lgb.train(
    params,
    train_data,
    valid_sets=[test_data]
)

# Make predictions
y_pred = np.argmax(model.predict(X_test, num_iteration=model.best_iteration), axis=1)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
