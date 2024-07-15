from flask import Flask, request, redirect, url_for, render_template, send_from_directory
import pandas as pd
import joblib
import os

app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Load the trained model
model_filename = 'rf_model.pkl'
rf_model = joblib.load(model_filename)

def price_recommendation(row):
    features = pd.DataFrame({
        'Present Price': [row['Present Price']],
        'Web Traffic': [row['Web Traffic']],
        'Units Sold': [row['Units Sold']],
        'Customer Ratings': [row['Customer Ratings']],
        'Stock Status': [row['Stock Status']]
    })

    # Encode the Stock Status
    stock_status_mapping = {'in stock': 0, 'out of product': 1}
    features['Stock Status'] = features['Stock Status'].map(stock_status_mapping)

    # Predict cart status
    predicted_cart_status = rf_model.predict(features)[0]

    # Map the predicted cart status back to the original labels
    cart_status_mapping = {0: 'incart', 1: 'purchased'}
    predicted_cart_status = cart_status_mapping[predicted_cart_status]

    # Recommend price adjustments based on predicted cart status
    if predicted_cart_status == 'purchased':
        recommended_price = row['Present Price'] * 1.10  # Increase price by 10% for purchased items
    else:
        recommended_price = row['Present Price'] * 0.90  # Decrease price by 10% for items in cart

    return predicted_cart_status, recommended_price

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(filepath)

        # Apply price recommendations
        df['Predicted Cart Status'], df['Recommended Price'] = zip(*df.apply(price_recommendation, axis=1))

        # Save the modified DataFrame to a new CSV file
        processed_filename = 'processed_' + file.filename
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        df.to_csv(processed_filepath, index=False)

        return render_template('result.html', filename=processed_filename)

@app.route('/processed/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if _name_ == '_main_':
    app.run(debug=True)