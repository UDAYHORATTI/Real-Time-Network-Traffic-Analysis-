# Real-Time-Network-Traffic-Analysis-
#Capture real-time network traffic using scapy or pcapy. Extract features like packet length, protocol, source/destination IPs, and packet inter-arrival times. Train a deep learning model (RNN) to detect anomalies and classify traffic as malicious or benign. Use real-time alerts for detecting intrusion attempts or anomalies in the network.
import numpy as np
import pandas as pd
import scapy.all as scapy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import time

# Function to capture network packets
def capture_packets(interface="eth0", packet_count=100):
    packets = scapy.sniff(iface=interface, count=packet_count)
    return packets

# Feature extraction from network packets
def extract_features(packets):
    features = []
    for packet in packets:
        packet_features = {
            'packet_length': len(packet),
            'time_to_live': packet.ttl if 'ttl' in packet else -1,
            'protocol': packet.proto,
            'src_ip': packet.src,
            'dst_ip': packet.dst,
            'packet_type': packet.type if 'type' in packet else -1
        }
        features.append(packet_features)
    return pd.DataFrame(features)

# Simulated labeled data for training (Malicious or Benign network traffic)
# In a real-world scenario, the data would be captured from the network
traffic_data = [
    {"packet_length": 500, "time_to_live": 128, "protocol": 6, "src_ip": "192.168.1.1", "dst_ip": "192.168.1.2", "packet_type": 1, "label": "Benign"},
    {"packet_length": 1500, "time_to_live": 64, "protocol": 17, "src_ip": "192.168.1.3", "dst_ip": "192.168.1.4", "packet_type": 2, "label": "Malicious"},
    {"packet_length": 80, "time_to_live": 128, "protocol": 6, "src_ip": "192.168.1.5", "dst_ip": "192.168.1.6", "packet_type": 1, "label": "Benign"},
    {"packet_length": 10000, "time_to_live": 32, "protocol": 6, "src_ip": "192.168.1.7", "dst_ip": "192.168.1.8", "packet_type": 1, "label": "Malicious"},
    {"packet_length": 120, "time_to_live": 128, "protocol": 1, "src_ip": "192.168.1.9", "dst_ip": "192.168.1.10", "packet_type": 1, "label": "Benign"}
]

# Convert traffic data into a DataFrame
df = pd.DataFrame(traffic_data)

# Convert categorical 'protocol' and 'packet_type' to numeric
df['protocol'] = df['protocol'].astype(float)
df['packet_type'] = df['packet_type'].astype(float)

# Label Encoding: Benign -> 0, Malicious -> 1
df['label'] = df['label'].map({'Benign': 0, 'Malicious': 1})

# Split into features and labels
X = df.drop(columns=['label'])
y = df['label']

# Feature Scaling: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the LSTM model for Intrusion Detection
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape the input data for LSTM
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"Model Accuracy: {accuracy[1]*100:.2f}%")

# Function to detect real-time network anomalies
def detect_intrusion(packets):
    # Extract features from captured packets
    packet_features = extract_features(packets)
    packet_features['protocol'] = packet_features['protocol'].astype(float)
    packet_features['packet_type'] = packet_features['packet_type'].astype(float)
    
    # Feature scaling
    packet_features_scaled = scaler.transform(packet_features.drop(columns=['src_ip', 'dst_ip']))
    
    # Reshape the data for LSTM
    packet_features_reshaped = np.reshape(packet_features_scaled, (packet_features_scaled.shape[0], packet_features_scaled.shape[1], 1))
    
    # Predict the class (malicious or benign)
    predictions = model.predict(packet_features_reshaped)
    predicted_classes = (predictions > 0.5).astype(int)
    
    for i, prediction in enumerate(predicted_classes):
        if prediction == 1:
            print(f"ALERT! Malicious network activity detected: {packets[i].summary()}")
        else:
            print(f"Normal activity: {packets[i].summary()}")

# Simulate real-time detection of network traffic
captured_packets = capture_packets(packet_count=5)
detect_intrusion(captured_packets)
