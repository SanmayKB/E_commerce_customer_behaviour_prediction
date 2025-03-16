import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "data/ecommerce_customer_data_custom_ratios.csv"
df = pd.read_csv(file_path)

# Selecting only the required columns
selected_features = ["Product Price", "Quantity", "Total Purchase Amount"]
target_column = "Product Category"

# Handle missing values
df = df.dropna(subset=[target_column] + selected_features)

# Feature Engineering (Adding More Non-Linearity)
df["Price_per_Unit"] = df["Total Purchase Amount"] / (df["Quantity"] + 1)  
df["Log_Total_Purchase"] = np.log1p(df["Total Purchase Amount"])

# Polynomial Features (Only Important Interactions)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
poly_features = poly.fit_transform(df[selected_features])
poly_feature_names = poly.get_feature_names_out(selected_features)
df_poly = pd.DataFrame(poly_features, columns=poly_feature_names)

# Concatenate original and polynomial features
df = pd.concat([df, df_poly], axis=1)

# Updated feature list
selected_features += ["Price_per_Unit", "Log_Total_Purchase"]
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

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define LightGBM Model with Tuned Parameters
model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="multiclass",
    num_class=len(df[target_column].unique()),
    learning_rate=0.07,
    n_estimators=800,
    max_depth=20,
    num_leaves=100,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.5,
    reg_alpha=1.5
)

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)
