import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="KNN Gender Prediction", layout="centered")

st.title("🛍️ Mall Customer Gender Prediction")
st.write("Using **K-Nearest Neighbors (KNN)** Algorithm")

# =========================
# Load Dataset
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("Mall_Customers.csv")
    return df

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())

# =========================
# Preprocessing
# =========================
df['Genre'] = df['Genre'].astype(str).str.strip().str.lower().map({
    'male': 0,
    'female': 1
})

X = df[['Annual Income (k$)', 'Age', 'Spending Score (1-100)']]
y = df['Genre']

# Handle missing values
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Train KNN Model
# =========================
knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='minkowski',
    p=2
)
knn.fit(X_train, y_train)

# =========================
# Model Evaluation
# =========================
y_pred = knn.predict(X_test)

st.subheader("📊 Model Performance")
st.write("**Accuracy:**", accuracy_score(y_test, y_pred))

st.write("**Confusion Matrix:**")
st.write(confusion_matrix(y_test, y_pred))

st.write("**Classification Report:**")
st.text(classification_report(y_test, y_pred))

# =========================
# User Input for Prediction
# =========================
st.subheader("🔮 Predict Gender")

income = st.number_input("Annual Income (k$)", min_value=0, max_value=200, value=50)
age = st.number_input("Age", min_value=10, max_value=100, value=30)
score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Predict"):
    input_data = np.array([[income, age, score]])
    input_data = scaler.transform(input_data)
    prediction = knn.predict(input_data)

    if prediction[0] == 0:
        st.success("🧑 Predicted Gender: **Male**")
    else:
        st.success("👩 Predicted Gender: **Female**")
