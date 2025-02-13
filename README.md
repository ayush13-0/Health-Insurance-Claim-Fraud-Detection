💳 Health Insurance Claim Fraud Detection
This project is a web application that predicts the likelihood of fraud in health insurance claims using machine learning models. It provides a user-friendly interface to input transaction details and get predictions from various models, including Random Forest, Logistic Regression, and XGBoost.

🚀 Features
User Input Interface: Enter transaction details to check for potential fraud.
Multiple Models: Choose between Random Forest, Logistic Regression, and XGBoost for predictions.
Real-Time Prediction: Get instant fraud prediction results.
Model Accuracy Display: Shows the accuracy of the selected model.
📊 Models Used
Random Forest Classifier
Logistic Regression
XGBoost Classifier
🔧 Installation
Clone the repository:
git clone https://github.com/yourusername/health-insurance-claim-fraud-detection.git
cd health-insurance-claim-fraud-detection

Create and activate a new conda environment:
conda create -n fraud-detection python=3.9
conda activate fraud-detection

Install the required dependencies:
pip install -r requirements.txt
⚙️ Usage
Ensure the dataset is present in the root directory as PS_20174392719_1491204439457_log.csv.

Run the application:
streamlit run fraud_detection_app.py
Access the web app in your browser at http://localhost:8501.

📁 Dataset
The dataset used in this project is named PS_20174392719_1491204439457_log.csv. It includes the following columns:

step: Time step of the transaction
type: Type of transaction (e.g., CASH-IN, CASH-OUT, TRANSFER, etc.)
amount: Transaction amount
nameOrig: Customer ID of the sender
oldbalanceOrg: Initial balance of the sender
newbalanceOrig: Balance of the sender after the transaction
nameDest: Customer ID of the receiver
oldbalanceDest: Initial balance of the receiver
newbalanceDest: Balance of the receiver after the transaction
isFraud: Indicator if the transaction is fraudulent (1 for fraud, 0 for non-fraud)
isFlaggedFraud: Transactions flagged as fraud by the system
🧑‍💻 Developed By
Ayush
