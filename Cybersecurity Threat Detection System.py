#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Simulating network traffic data
np.random.seed(42)
normal_data = np.random.normal(loc=50, scale=10, size=(1000, 4))  # Normal network traffic
anomaly_data = np.random.normal(loc=100, scale=20, size=(50, 4))  # Anomalous network traffic


# In[3]:


# Combine both normal and anomalous data
traffic_data = np.vstack([normal_data, anomaly_data])
labels = np.hstack([np.zeros(1000), np.ones(50)])  # 0 = normal, 1 = anomaly


# In[4]:


# Create a DataFrame for analysis
columns = ['packet_size', 'duration', 'source_port', 'destination_port']
df = pd.DataFrame(traffic_data, columns=columns)
df['label'] = labels

print(df.head())


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split data into features and labels
X = df.drop('label', axis=1)  # Features
y = df['label']  # Labels (0 = normal, 1 = anomaly)

# Split into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print classification report and confusion matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[8]:


import matplotlib.pyplot as plt

# Plot feature importance
importances = rf_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(X_train.shape[1]), importances[indices], align='center')
plt.yticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()



# In[9]:


# Example of how to detect anomalies (new unseen data)
new_data = np.array([[60, 30, 5000, 80]])  # Example of a new traffic packet
new_data_scaled = scaler.transform(new_data)

# Predict if it's an anomaly (1) or normal (0)
prediction = rf_model.predict(new_data_scaled)
if prediction[0] == 1:
    print("Anomaly detected!")
else:
    print("Traffic is normal.")


# In[ ]:




