import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report


file_path = "data/ecommerce_customer_data_custom_ratios.csv"  
df = pd.read_csv(file_path)


features = [ "Product Price", "Quantity", "Total Purchase Amount", 
            "Payment Method", "Returns"]
target = "Product Category"


df = df.dropna(subset=[target] + features)


label_encoders = {}
for col in ["Payment Method"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  


target_encoder = LabelEncoder()
df[target] = target_encoder.fit_transform(df[target])


X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = xgb.XGBClassifier(
    objective="multi:softmax",
    num_class=len(df[target].unique()),
    learning_rate=0.1,
    n_estimators=200,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_encoder.classes_)

print(f"Accuracy: {accuracy:.4f}")
print("Detailed Classification Report:")
print(report)
