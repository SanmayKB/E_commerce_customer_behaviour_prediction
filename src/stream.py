import streamlit as st
import joblib
import pandas as pd


# title
st.title("Customer Behavior Prediction")

# making the input fields here
st.header("Select the ML model")
modelType = st.selectbox("Model",["Random Forest","XGBoost","Neural Network"])
st.divider()
customer_id = st.number_input("Customer ID", min_value=0, step=1)
age = st.number_input("Age",min_value = 0, step = 1)
gender = st.selectbox("Gender", ["Male", "Female"])
product_category = st.selectbox("Product Category",["Books","Clothing","Electronics","Home"])
product_price = st.number_input("Product Price", min_value=0.0, step=0.01)
quantity = st.number_input("Quantity", min_value=1, step=1)
payment_method = st.selectbox("Payment Method",["Cash","Credit Card", "Crypto", "Paypal"])
returns = st.selectbox("Returns", ["Yes", "No"])  

#based on the user's choice we can select the model we wish to predict with
if modelType == "Random Forest":
    model = joblib.load("src/models/random_forest_model.pkl")
elif modelType == "XGBoost":
    model = joblib.load("src/models/xgboost_model.pkl")
elif modelType == "Neural Network":
    model = joblib.load("src/models/neural_network_model.pkl")
    


books = clothing = electronics = home = 0
if product_category=="Books":
    books = 1
elif product_category == "Clothing":
    clothing = 1
elif product_category == "Electronics":
    electronics = 1
else:
    home = 1

cash = creditCard = crypto = paypal = 0
if payment_method == "Cash":
    cash = 1
elif payment_method == "Credit Card":
    creditCard = 1
elif payment_method == "Crypto":
    crypto = 1 
else:
    paypal = 1

val = 0
if returns == "Yes":
    val = 1
    
    
gend = 0
if gender == "Male":
    gend = 1    
    

if st.button("Predict"):
    
    data = {
        "Quantity": quantity,
        "Customer Age":age,
        "Returns": val,
        "Gender": gend,
        'Product Category_Books':books, 
        'Product Category_Clothing':clothing,
        'Product Category_Electronics':electronics, 
        'Product Category_Home':home,
        'Payment Method_Cash':cash, 
        'Payment Method_Credit Card':creditCard,
        'Payment Method_Crypto':crypto, 
        'Payment Method_PayPal':paypal
        
    }
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    #get our prediction
    prediction = model.predict(df)[0]
    
    
    #output our prediction
    if(int(prediction)==1):
        st.success(f"The Customer is likely to be a returning customer. Predicted Churn: {int(prediction)}")
    else:
        st.success(f"The Customer is not likely to be a returning customer. Predicted Churn: {int(prediction)}")
