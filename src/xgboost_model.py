import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
file_path = "data/ecommerce_customer_data_custom_ratios.csv"  # Update this if needed
df = pd.read_csv(file_path)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Ensure 'Product Category' is the target variable
target_column = "Product Category"
if target_column not in df.columns:
    raise ValueError("Target column 'Product Category' not found in dataset.")

# Remove target column from features
if target_column in numerical_cols:
    numerical_cols.remove(target_column)

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Handle missing values
df = df.dropna(subset=[target_column] + numerical_cols + categorical_cols)

# Feature Engineering: Create Interaction Terms
df["Price_Quantity_Interaction"] = df["Product Price"] * df["Quantity"]
df["Total_Price_Ratio"] = df["Total Purchase Amount"] / (df["Product Price"] + 1)  # Avoid division by zero

# Standardize numerical features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Encode target variable
label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])

# Define Features and Target
X = df[numerical_cols + categorical_cols + ["Price_Quantity_Interaction", "Total_Price_Ratio"]]
y = df[target_column]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(df[target_column].unique()),
    learning_rate=0.05,
    n_estimators=700,
    max_depth=20,
    min_child_weight=3,
    gamma=0.2,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=0.8,
    reg_alpha=0.8,
    tree_method="hist",  # Faster training
    verbosity=1
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Bar Plot for Prediction Distribution
plt.figure(figsize=(10, 5))
sns.barplot(x=label_encoder.classes_, y=pd.Series(y_pred).value_counts().sort_index(), palette="viridis")
plt.xlabel("Product Category")
plt.ylabel("Prediction Count")
plt.title("Distribution of Predicted Product Categories")
plt.xticks(rotation=45)
plt.show()
