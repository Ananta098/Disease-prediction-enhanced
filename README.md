# Disease-prediction-enhanced

An intelligent healthcare application that leverages advanced machine learning algorithms to predict potential diseases based on user-reported symptoms. Built with Python, scikit-learn, and Flask, this system provides accurate disease predictions along with detailed health recommendations.

## Key Features

- **Smart Symptom Analysis**: Advanced NLP-based symptom matching using fuzzy logic and semantic similarity
- **High Accuracy Predictions**: Achieves accuracy using ensemble machine learning models (XGBoost, Random Forest, LightGBM)
- **Multiple Prediction Support**: Returns top 3 most likely diagnoses with confidence scores
- **Comprehensive Health Information**: Provides precautions, medications, dietary recommendations, and exercise guidelines
- **Modern Web Interface**: Beautiful, responsive UI built with Tailwind CSS
- **Flexible Input**: Accepts various symptom descriptions (handles typos, synonyms, and paraphrasing)
- **Real-time Processing**: Instant predictions with interactive feedback

## Technology Stack

- **Backend**: Python 3.10+, Flask
- **Machine Learning**: scikit-learn, XGBoost, LightGBM, NumPy, Pandas
- **NLP**: NLTK, FuzzyWuzzy, spaCy (optional)
- **Frontend**: HTML5, Tailwind CSS, Vanilla JavaScript
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: joblib

## Model Performance

- **Dataset**: 773 diseases, 377 symptoms, 246,000+ samples
- **Accuracy**: 95-98% on test data
- **Models Used**: XGBoost, Random Forest, LightGBM (ensemble approach)
- **Preprocessing**: Advanced text normalization, lemmatization, feature engineering

## Use Cases

- Educational tool for medical students
- Preliminary symptom checker for patients
- Healthcare research and analysis
- Telemedicine support system
- Medical AI demonstrations

## Disclaimer

This system is designed for educational and informational purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical concerns.

## Project Highlights

- Production-ready code with comprehensive error handling
- Modular architecture for easy maintenance and scaling
- Extensive test suite with 95%+ coverage
- Well-documented codebase following Python best practices
- RESTful API for easy integration

## Learning Outcomes

This project demonstrates:
- End-to-end machine learning pipeline
- NLP and text processing techniques
- Web application development
- Model evaluation and optimization
- Healthcare AI applications
