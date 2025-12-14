from flask import Flask, render_template, request, jsonify
import sys
sys.path.append('..')
from src.models.predict import DiseasePredictor
import json

app = Flask(__name__)

# Initialize predictor
predictor = DiseasePredictor(
    model_path='../models/best_model.pkl',
    vocab_path='../data/processed/vocabulary.pkl'
)

# Load additional information (you'll need to create this)
with open('../data/disease_info.json', 'r') as f:
    disease_info = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        user_symptoms = data.get('symptoms', [])
        
        if not user_symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400
        
        # Get predictions
        predictions, matched_symptoms = predictor.predict(user_symptoms, return_top_n=3)
        
        if not predictions:
            return jsonify({
                'error': 'Could not match symptoms. Please try different descriptions.',
                'matched_symptoms': []
            }), 404
        
        # Format response
        results = []
        for disease, probability, _ in predictions:
            disease_data = disease_info.get(disease, {})
            results.append({
                'disease': disease,
                'probability': float(probability),
                'description': disease_data.get('description', ''),
                'precautions': disease_data.get('precautions', []),
                'medications': disease_data.get('medications', []),
                'workout': disease_data.get('workout', []),
                'diet': disease_data.get('diet', [])
            })
        
        return jsonify({
            'predictions': results,
            'matched_symptoms': [{'symptom': s, 'confidence': c} for s, c in matched_symptoms]
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/suggest_symptoms', methods=['POST'])
def suggest_symptoms():
    try:
        data = request.json
        partial = data.get('partial', '')
        
        suggestions = predictor.get_symptom_suggestions(partial)
        
        return jsonify({'suggestions': suggestions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)