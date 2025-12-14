import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.data.preprocessor import DiseaseDataPreprocessor

print("="*60)
print("REGENERATING VOCABULARY FILE")
print("="*60)

# Check if processed data exists
processed_data_path = 'data/processed/processed_disease_data.csv'

if os.path.exists(processed_data_path):
    print(f"\n✓ Found processed data: {processed_data_path}")
    
    # Load processed data
    df = pd.read_csv(processed_data_path)
    print(f"  Loaded {len(df)} samples")
    
    # Initialize preprocessor
    preprocessor = DiseaseDataPreprocessor()
    
    # Recreate vocabulary from processed data
    # Assuming symptoms are stored as lists in a column
    if 'symptoms' in df.columns:
        for idx, row in df.iterrows():
            symptoms = eval(row['symptoms']) if isinstance(row['symptoms'], str) else row['symptoms']
            for symptom in symptoms:
                preprocessor.symptom_vocabulary.add(symptom)
    
    # Recreate label encoder
    if 'disease' in df.columns:
        preprocessor.label_encoder.fit(df['disease'])
    
    print(f"\n✓ Vocabulary recreated:")
    print(f"  - {len(preprocessor.symptom_vocabulary)} unique symptoms")
    print(f"  - {len(preprocessor.label_encoder.classes_)} unique diseases")
    
    # Save vocabulary
    output_path = 'data/processed/vocabulary.pkl'
    preprocessor.save_vocabulary(output_path)
    print(f"\n✓ Vocabulary saved to: {output_path}")
    
else:
    print(f"\n❌ Error: {processed_data_path} not found")
    print("\nYou need to run the preprocessing notebook first:")
    print("  jupyter notebook notebooks/02_data_preprocessing.ipynb")

print("="*60)