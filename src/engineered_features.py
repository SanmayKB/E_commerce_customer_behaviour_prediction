import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report


file_path = "data/ecommerce_customer_data_custom_ratios.csv"
df = pd.read_csv(file_path)


selected_features = ["Product Price", "Quantity", "Total Purchase Amount"]
target_column = "Product Category"


df = df.dropna(subset=[target_column] + selected_features)


label_encoder = LabelEncoder()
df[target_column] = label_encoder.fit_transform(df[target_column])


X = df[selected_features]
y = df[target_column]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lgb_model = lgb.LGBMClassifier(
    boosting_type="gbdt",
    objective="multiclass",
    num_class=len(df[target_column].unique()),
    learning_rate=0.05,
    n_estimators=500,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=1.0
)
lgb_model.fit(X_train_scaled, y_train)

# this is used to check the weight of the features
feature_importances = lgb_model.feature_importances_
feature_names = X_train.columns

plt.figure(figsize=(10, 5))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance (LightGBM)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()


xgb_model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(df[target_column].unique()),
    learning_rate=0.05,
    n_estimators=500,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=1.0
)
xgb_model.fit(X_train_scaled, y_train)


y_pred_lgb = lgb_model.predict(X_test_scaled)
y_pred_xgb = xgb_model.predict(X_test_scaled)

accuracy_lgb = accuracy_score(y_test, y_pred_lgb)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

print(f"LightGBM Accuracy: {accuracy_lgb:.4f}")
print("LightGBM Classification Report:")
print(classification_report(y_test, y_pred_lgb))

print(f"\nXGBoost Accuracy: {accuracy_xgb:.4f}")
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
