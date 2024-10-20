# Import necessary libraries
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random
import os

# Load the training (September) and validation (May-December) data
train_data = pd.read_csv('Sep_out.csv')
val_data = pd.read_csv('5-12 out.csv')

# Data Preprocessing Function
def preprocess_data(data, date_col, quantity_col, customer_col, customer_name_col):
    data.rename(columns=lambda x: x.strip(), inplace=True)
    data.rename(columns={date_col: 'date', quantity_col: 'quantity_out', customer_col: 'customer_id', customer_name_col: 'customer_name'}, inplace=True)
    data.dropna(subset=['date', 'quantity_out'], inplace=True)
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data.dropna(subset=['date'], inplace=True)
    return data

# Preprocess training and validation data
train_data = preprocess_data(train_data, train_data.columns[2], train_data.columns[25], train_data.columns[7], train_data.columns[8])
val_data = preprocess_data(val_data, val_data.columns[2], val_data.columns[25], val_data.columns[7], val_data.columns[6])

# Function to create time series dataset
def create_dataset(data, time_step=5):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Placeholder for anomaly detection results
anomalies = []
customer_predictions = {}

# Directory to save models
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# Loop through each customer and build a model using training data, then validate using validation data
for customer_id, customer_train_data in train_data.groupby('customer_id'):
    if customer_train_data.empty:
        continue

    customer_name = customer_train_data['customer_name'].iloc[0]
    customer_train_data = customer_train_data.sort_values('date')
    train_quantities = customer_train_data[['quantity_out']].values

    # Scale the training data
    scaled_train_data = scaler.fit_transform(train_quantities)

    # Create the training dataset
    time_step = 5
    X_train, y_train = create_dataset(scaled_train_data, time_step)
    if len(X_train) == 0 or len(y_train) == 0:
        continue

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Adjust batch size if larger than training set
    batch_size = min(32, len(X_train))

    # Model file path
    model_path = os.path.join(model_dir, f'model_{customer_id}.h5')

    # Load or build LSTM model
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, batch_size=batch_size, epochs=10)
        model.save(model_path)

    # Validate the model using validation data for the same customer
    if customer_id in val_data['customer_id'].unique():
        customer_val_data = val_data[val_data['customer_id'] == customer_id].sort_values('date')
        val_quantities = customer_val_data[['quantity_out']].values

        # Scale the validation data
        scaled_val_data = scaler.transform(val_quantities)

        # Create the validation dataset
        X_val, y_val = create_dataset(scaled_val_data, time_step)
        if len(X_val) == 0 or len(y_val) == 0:
            continue

        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)

        # Make predictions
        predictions = model.predict(X_val)
        predictions = scaler.inverse_transform(predictions)
        y_val_scaled = scaler.inverse_transform(y_val.reshape(-1, 1))

        # Store predictions for visualization
        customer_predictions[customer_name] = (y_val_scaled, predictions)

        # Calculate residuals and detect anomalies
        residuals = predictions - y_val_scaled
        threshold = np.mean(np.abs(residuals)) + 2 * np.std(residuals)  # Set threshold for anomaly detection
        anomaly_indices = np.where(np.abs(residuals) > threshold)[0]

        # Store anomalies with customer name and date
        for idx in anomaly_indices:
            anomalies.append({
                'customer_name': customer_name,
                'date': customer_val_data.iloc[time_step + idx]['date'],
                'actual_quantity': y_val_scaled[idx][0],
                'predicted_quantity': predictions[idx][0],
                'difference': residuals[idx][0]
            })

# Convert anomalies to DataFrame and save to CSV
anomalies_df = pd.DataFrame(anomalies)
anomalies_df.to_csv('customer_anomalies_report.csv', index=False)

# Separate anomalies into positive and negative changes
anomalies_df['difference'] = anomalies_df['predicted_quantity'] - anomalies_df['actual_quantity']
positive_changes = anomalies_df[anomalies_df['difference'] > 0]
negative_changes = anomalies_df[anomalies_df['difference'] < 0]

# Select top 10 customers with the largest positive changes in quantity
top_10_positive = positive_changes.groupby('customer_name')['difference'].mean().nlargest(10).reset_index()

# Select top 10 customers with the largest negative changes in quantity
top_10_negative = negative_changes.groupby('customer_name')['difference'].mean().nsmallest(10).reset_index()

# Set up the font to support Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题

# Plot top 10 customers with the largest positive changes in quantity
plt.figure(figsize=(12, 8))
plt.barh(top_10_positive['customer_name'], top_10_positive['difference'], color='green')
plt.xlabel('Average Quantity Increase')
plt.title('Top 10 Customers with Largest Increases in Quantity')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Plot top 10 customers with the largest negative changes in quantity
plt.figure(figsize=(12, 8))
plt.barh(top_10_negative['customer_name'], top_10_negative['difference'], color='red')
plt.xlabel('Average Quantity Decrease')
plt.title('Top 10 Customers with Largest Decreases in Quantity')
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

