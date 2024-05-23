# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from flask import Flask, request, jsonify

# Load the dataset
dataset_url = "https://www.narcis.nl/dataset/RecordID/oai%3Aeasy.dans.knaw.nl%3Aeasy-dataset%3A191591"
# Download the dataset manually and load it using pandas
# df = pd.read_csv("path_to_downloaded_dataset.csv")

# For demonstration purposes, let's create a dummy DataFrame
# Replace this with actual dataset loading code
df = pd.DataFrame({'Type_of_collision': [1, 2, 3, 4, 5],
                   'Cause_of_accident': [0.1, 0.2, 0.3, 0.4, 0.5],
                   'Accident_severity': [0, 1, 0, 1, 1]})

# Preprocess the dataset
X = df.drop('Accident_severity', axis=1)
y = df['Accident_severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Initialize Flask app
app = Flask(__name__)

# Define endpoint for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = data['Type_of_collision','Cause_of_accident','Accident_severity']
    features = [features['Type_of_collision'], features['Cause_of_accident']]  # Assuming feature1 and feature2 are present
    prediction = model.predict([features])[0]
    return jsonify({'prediction': prediction})

if __name__ == '_main_':
    app.run(debug=True)
