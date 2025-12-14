import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class DiseaseDataPreprocessor:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.symptom_vocabulary = set()
        
    def clean_symptom_text(self, text):
        """Clean and normalize symptom text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    def prepare_dataset(self, df, symptom_columns):
        """
        Prepare dataset for training
        df: DataFrame with Disease and symptom columns
        symptom_columns: List of column names containing symptoms
        """
        processed_data = []
        
        for idx, row in df.iterrows():
            disease = row['Disease']
            symptoms = []
            
            # Collect all symptoms for this row
            for col in symptom_columns:
                symptom = row[col]
                if pd.notna(symptom) and str(symptom).strip():
                    cleaned = self.clean_symptom_text(symptom)
                    if cleaned:
                        symptoms.append(cleaned)
                        self.symptom_vocabulary.add(cleaned)
            
            if symptoms:  # Only add if there are symptoms
                processed_data.append({
                    'disease': disease,
                    'symptoms': symptoms
                })
        
        return pd.DataFrame(processed_data)
    
    def create_feature_matrix(self, df):
        """
        Create binary feature matrix
        Each row is a sample, each column is a symptom
        """
        # Create symptom to index mapping
        symptom_list = sorted(list(self.symptom_vocabulary))
        symptom_to_idx = {s: i for i, s in enumerate(symptom_list)}
        
        # Create feature matrix
        X = np.zeros((len(df), len(symptom_list)))
        
        for idx, row in df.iterrows():
            for symptom in row['symptoms']:
                if symptom in symptom_to_idx:
                    X[idx, symptom_to_idx[symptom]] = 1
        
        # Encode target variable
        y = self.label_encoder.fit_transform(df['disease'])
        
        return X, y, symptom_list
    
    def save_vocabulary(self, filepath):
        """Save symptom vocabulary for later use"""
        import joblib
        joblib.dump({
            'symptoms': list(self.symptom_vocabulary),
            'label_encoder': self.label_encoder
        }, filepath)
    
    def load_vocabulary(self, filepath):
        """Load saved vocabulary"""
        import joblib
        data = joblib.load(filepath)
        self.symptom_vocabulary = set(data['symptoms'])
        self.label_encoder = data['label_encoder']