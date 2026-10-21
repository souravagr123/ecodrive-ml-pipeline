from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
# Enable CORS so the HTML file can communicate with this Python server
CORS(app) 

# 1. Load the trained machine learning artifacts
print("Loading ML Models...")
try:
    preprocessor = joblib.load('preprocessor.pkl')
    rfe_clf = joblib.load('rfe_clf.pkl')
    rfe_reg = joblib.load('rfe_reg.pkl')
    best_rf_clf = joblib.load('best_rf_clf.pkl')
    best_rf_reg = joblib.load('best_rf_reg.pkl')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models. Ensure all 5 .pkl files are in this directory. Details: {e}")

# 2. Serve the HTML Dashboard Interface
@app.route('/')
def home():
    return render_template('index.html')

# 3. Prediction Endpoint for Frontend Form
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive data from the HTML frontend
        data = request.json
        df = pd.DataFrame([data])
        
        # Apply Feature Engineering (Energy Efficiency Index)
        df['Energy_Efficiency_Index'] = (
            df['Max_Range_km'] / df['Battery_Capacity_kWh']
        ) * (1 - (df['Vehicle_Age_Years'] * 0.02))
        
        # Capture EEI to send back to the UI
        calculated_eei = float(df['Energy_Efficiency_Index'].iloc[0])
        
        # Preprocess Data (Scaling & Encoding)
        X_processed = preprocessor.transform(df)
        
        # Apply Feature Selection
        X_selected_class = rfe_clf.transform(X_processed)
        X_selected_reg = rfe_reg.transform(X_processed)
        
        # Generate Predictions using Ensembles
        pred_liquidity = int(best_rf_clf.predict(X_selected_class)[0])
        pred_price = float(best_rf_reg.predict(X_selected_reg)[0])
        
        # Return Results to Frontend
        return jsonify({
            'status': 'success',
            'liquidity_prediction': pred_liquidity,
            'price_prediction': pred_price,
            'engineered_eei': calculated_eei
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)