import joblib
import numpy as np
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Now import with relative path
from src.nlp.symptom_matcher import SymptomMatcher
from src.data.preprocessor import DiseaseDataPreprocessor

class DiseasePredictor:
    def __init__(self, model_path, vocab_path):
        # Load model
        self.model = joblib.load(model_path)
        
        # Load vocabulary and encoder
        vocab_data = joblib.load(vocab_path)
        self.symptom_list = vocab_data['symptoms']
        self.label_encoder = vocab_data['label_encoder']
        
        # Initialize symptom matcher
        self.matcher = SymptomMatcher(self.symptom_list)
        
        # Create symptom to index mapping
        self.symptom_to_idx = {s: i for i, s in enumerate(self.symptom_list)}
    
    def predict(self, user_symptoms, return_top_n=3, confidence_threshold=0.3):
        """
        Predict disease based on user symptoms
        
        Args:
            user_symptoms: List of symptom strings from user
            return_top_n: Number of top predictions to return
            confidence_threshold: Minimum confidence for predictions
        
        Returns:
            List of (disease, probability, matched_symptoms) tuples
        """
        # Match user symptoms to vocabulary
        matched_symptoms = self.matcher.match_multiple_symptoms(
            user_symptoms, threshold=75
        )
        
        if not matched_symptoms:
            return [], []
        
        # Create feature vector
        feature_vector = np.zeros((1, len(self.symptom_list)))
        used_symptoms = []
        
        for symptom, confidence in matched_symptoms:
            if symptom in self.symptom_to_idx:
                feature_vector[0, self.symptom_to_idx[symptom]] = 1
                used_symptoms.append((symptom, confidence))
        
        # Predict
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Get top predictions
        top_indices = np.argsort(probabilities)[::-1][:return_top_n]
        
        predictions = []
        for idx in top_indices:
            prob = probabilities[idx]
            if prob >= confidence_threshold:
                disease = self.label_encoder.inverse_transform([idx])[0]
                predictions.append((disease, prob, used_symptoms))
        
        return predictions, used_symptoms
    
    def get_symptom_suggestions(self, partial_input):
        """Get symptom suggestions for autocomplete"""
        return self.matcher.suggest_symptoms(partial_input, top_n=10)