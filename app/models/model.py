import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MLModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        """Train the model on the given data"""
        self.model.fit(X, y)
        
    def predict(self, X):
        """Make predictions with the model"""
        return self.model.predict(X)
    
    def save(self, filename):
        """Save the model to a file"""
        joblib.dump(self.model, filename)
        
    def load(self, filename):
        """Load the model from a file"""
        self.model = joblib.load(filename)