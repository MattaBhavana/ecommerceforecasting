import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('/content/ecommerce_dataset_int_price.csv')

# Preprocess the data
features = df[['Present Price', 'Web Traffic', 'Units Sold', 'Customer Ratings', 'Stock Status']]
target = df['Cart Status']

# Encode categorical features
label_encoder = LabelEncoder()
features['Stock Status'] = label_encoder.fit_transform(features['Stock Status'])
target = label_encoder.fit_transform(target)

# Convert target to pandas Series
target = pd.Series(target)

# Split the data into training and testing sets with stratification
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss.split(features, target):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target.iloc[train_index], target.iloc[test_index]

# Build and train the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate the model
y_pred = rf_model.predict(X_test)
unique_classes_in_test = np.unique(y_pred)
target_names_in_test = [label_encoder.classes_[i] for i in unique_classes_in_test]
print(classification_report(y_test, y_pred, target_names=target_names_in_test))

# Predict sales categories for the entire dataset
df['Predicted Cart Status'] = rf_model.predict(features)
df['Predicted Cart Status'] = label_encoder.inverse_transform(df['Predicted Cart Status'])

# Recommend price adjustments
def price_recommendation(row):
    if row['Predicted Cart Status'] == 'purchased':
        return row['Present Price'] * 1.10  # Increase price by 10% for purchased items
    else:
        return row['Present Price'] * 0.90  # Decrease price by 10% for items in cart

df['Recommended Price'] = df.apply(price_recommendation, axis=1)

# Display the recommendations
print(df[['Product ID', 'Units Sold', 'Cart Status', 'Present Price', 'Predicted Cart Status', 'Recommended Price','Customer Ratings','Stock Status']])

# Optionally, save the results to a CSV file
df.to_csv('/content/price_recommendations.csv', index=False)

import joblib
model_path = '/content/rf_model.pkl'
joblib.dump(rf_model, 'rf_model.pkl')