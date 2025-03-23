import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import lightgbm as lgb


file_path = "data/ecommerce_customer_data_custom_ratios.csv"
df = pd.read_csv(file_path)


df.drop(columns=["Customer Name", "Age", "Customer ID", "Purchase Date"], errors='ignore', inplace=True)


df.fillna(df.median(numeric_only=True), inplace=True)


categorical_cols = ["Payment Method", "Gender"]
df = pd.get_dummies(df, columns=categorical_cols)


if "Total Purchase Amount" in df.columns:
    df["Total Purchase Amount"] = np.log1p(df["Total Purchase Amount"] + 1)


label_encoder = LabelEncoder()
df["Product Category"] = label_encoder.fit_transform(df["Product Category"])


X = df.drop(columns=["Product Category"], errors='ignore')  # Features
y = df["Product Category"]  # Target


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


if X_scaled.shape[1] > 10:
    pca = PCA(n_components=0.95)
    X_reduced = pca.fit_transform(X_scaled)
else:
    X_reduced = X_scaled  


X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


model = lgb.LGBMClassifier(
    n_estimators=800,       
    learning_rate=0.08,     
    max_depth=14,           
    num_leaves=70,          
    colsample_bytree=0.9,   
    subsample=0.85,         
    reg_alpha=0.05,         
    reg_lambda=0.05,        
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred))

#The graph for visualisation of how important each feature is
"""
importances = model.feature_importances_
plt.figure(figsize=(10, 5))
plt.barh(range(len(importances)), importances)
plt.xlabel("Importance")
plt.ylabel("Feature Index")
plt.title("Feature Importance")
plt.show()

"""
