import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler


file_path = "data/ecommerce_customer_data_custom_ratios.csv"  
df = pd.read_csv(file_path)


categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()


target_column = "Product Category"
if target_column not in df.columns:
    raise ValueError("Target column 'Product Category' not found in dataset.")


if target_column in numerical_cols:
    numerical_cols.remove(target_column)


label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


df = df.dropna(subset=[target_column] + numerical_cols + categorical_cols)


df["Price_Quantity_Interaction"] = df["Product Price"] * df["Quantity"]
df["Total_Price_Ratio"] = df["Total Purchase Amount"] / (df["Product Price"] + 1)  # Avoid division by zero


scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])


X = df[numerical_cols + categorical_cols + ["Price_Quantity_Interaction", "Total_Price_Ratio"]]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


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
    tree_method="hist",  
    verbosity=1
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)


plt.figure(figsize=(10, 5))
sns.barplot(x=label_encoder.classes_, y=pd.Series(y_pred).value_counts().sort_index(), palette="viridis")
plt.xlabel("Product Category")
plt.ylabel("Prediction Count")
plt.title("Distribution of Predicted Product Categories")
plt.xticks(rotation=45)
plt.show()
