import unittest
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.predict import DiseasePredictor

class TestDiseasePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures"""
        print("\n" + "="*60)
        print("Initializing TestDiseasePredictor")
        print("="*60)
        
        model_path = os.path.join(project_root, 'models/best_model.pkl')
        vocab_path = os.path.join(project_root, 'data/processed/vocabulary.pkl')
        
        cls.predictor = DiseasePredictor(
            model_path=model_path,
            vocab_path=vocab_path
        )
        
        # Get actual symptoms from vocabulary for testing
        cls.actual_symptoms = list(cls.predictor.symptom_list)[:20]  # Get 20 symptoms
        print(f"\n✓ Loaded {len(cls.predictor.symptom_list)} symptoms")
        print(f"✓ Loaded {len(cls.predictor.label_encoder.classes_)} diseases")
        print(f"\nUsing these symptoms for testing: {cls.actual_symptoms[:5]}")
        print("="*60 + "\n")
    
    def test_exact_symptom_match(self):
        """Test with exact symptom names from vocabulary"""
        print("\n📝 Test: Exact symptom match")
        
        # Use 7 symptoms for better confidence
        test_symptoms = self.actual_symptoms[:7]
        print(f"   Input: {test_symptoms[:3]}... ({len(test_symptoms)} total)")
        
        # Use lower threshold for testing
        predictions, matched = self.predictor.predict(
            test_symptoms, 
            confidence_threshold=0.05
        )
        
        print(f"   Matched: {len(matched)} symptoms")
        print(f"   Predictions: {len(predictions)}")
        
        if predictions:
            for disease, prob, _ in predictions[:3]:
                print(f"     - {disease}: {prob:.2%}")
        
        self.assertGreater(len(predictions), 0, "Should return predictions for exact matches")
        self.assertEqual(len(matched), len(test_symptoms), "Should match all exact symptoms")
        print("   ✓ Test passed\n")
    
    def test_fuzzy_symptom_match(self):
        """Test with similar symptom names"""
        print("\n📝 Test: Fuzzy symptom match")
        
        # Use variations of actual symptoms
        if len(self.actual_symptoms) > 0:
            base_symptom = self.actual_symptoms[0]
            
            variations = [
                base_symptom,
                base_symptom + 's',
                base_symptom.replace('_', ' '),
            ]
            
            print(f"   Input variations: {variations[:2]}")
            
            predictions, matched = self.predictor.predict(
                variations,
                confidence_threshold=0.05
            )
            
            print(f"   Matched: {len(matched)} symptoms")
            print(f"   Predictions: {len(predictions)}")
            
            self.assertGreater(len(matched), 0, "Should match similar symptoms")
            print("   ✓ Test passed\n")
    
    def test_invalid_symptoms(self):
        """Test with completely invalid symptoms"""
        print("\n📝 Test: Invalid symptoms")
        
        symptoms = ['xyz123', 'abcdef', 'notasymptom']
        print(f"   Input: {symptoms}")
        
        predictions, matched = self.predictor.predict(symptoms)
        
        print(f"   Matched: {len(matched)} symptoms")
        print(f"   Predictions: {len(predictions)}")
        
        # Should match 0 symptoms
        self.assertEqual(len(matched), 0, "Should not match invalid symptoms")
        print("   ✓ Test passed\n")
    
    def test_mixed_symptoms(self):
        """Test with mix of valid and invalid symptoms"""
        print("\n📝 Test: Mixed valid/invalid symptoms")
        
        # Use more valid symptoms
        valid_symptoms = self.actual_symptoms[:5]  # Increased to 5
        invalid_symptoms = ['xyz123', 'notreal']
        mixed = valid_symptoms + invalid_symptoms
        
        print(f"   Valid: {valid_symptoms[:2]}... ({len(valid_symptoms)} total)")
        print(f"   Invalid: {invalid_symptoms}")
        
        predictions, matched = self.predictor.predict(
            mixed,
            confidence_threshold=0.05
        )
        
        print(f"   Matched: {len(matched)} symptoms")
        print(f"   Predictions: {len(predictions)}")
        
        if predictions:
            for disease, prob, _ in predictions[:3]:
                print(f"     - {disease}: {prob:.2%}")
        
        self.assertGreater(len(predictions), 0, "Should return predictions for valid symptoms")
        self.assertEqual(len(matched), len(valid_symptoms), 
                        f"Should match {len(valid_symptoms)} valid symptoms")
        print("   ✓ Test passed\n")
    
    def test_autocomplete(self):
        """Test symptom suggestions"""
        print("\n📝 Test: Autocomplete suggestions")
        
        if len(self.actual_symptoms) > 0:
            partial = self.actual_symptoms[0][:3]
            print(f"   Partial input: '{partial}'")
            
            suggestions = self.predictor.get_symptom_suggestions(partial)
            
            print(f"   Suggestions: {suggestions[:5]}")
            
            self.assertGreater(len(suggestions), 0, "Should return suggestions")
            print("   ✓ Test passed\n")
    
    def test_prediction_confidence(self):
        """Test that predictions have reasonable confidence scores"""
        print("\n📝 Test: Prediction confidence")
        
        # Use enough symptoms for good predictions
        test_symptoms = self.actual_symptoms[:8]
        predictions, matched = self.predictor.predict(
            test_symptoms,
            confidence_threshold=0.05
        )
        
        if predictions:
            for disease, prob, _ in predictions[:3]:
                print(f"   {disease}: {prob:.2%}")
                self.assertGreater(prob, 0, "Probability should be > 0")
                self.assertLessEqual(prob, 1, "Probability should be <= 1")
        
        print("   ✓ Test passed\n")
    
    def test_multiple_predictions(self):
        """Test that we get multiple disease predictions"""
        print("\n📝 Test: Multiple predictions")
        
        # Use enough symptoms
        test_symptoms = self.actual_symptoms[:10]
        predictions, matched = self.predictor.predict(
            test_symptoms, 
            return_top_n=5,
            confidence_threshold=0.05
        )
        
        print(f"   Input symptoms: {len(test_symptoms)}")
        print(f"   Top predictions: {len(predictions)}")
        
        if predictions:
            for i, (disease, prob, _) in enumerate(predictions, 1):
                print(f"   {i}. {disease}: {prob:.2%}")
        
        self.assertGreater(len(predictions), 0, "Should return at least 1 prediction")
        print("   ✓ Test passed\n")
    
    def test_realistic_scenario(self):
        """Test with a realistic number of symptoms"""
        print("\n📝 Test: Realistic scenario (5-8 symptoms)")
        
        # Most users will enter 5-8 symptoms
        test_symptoms = self.actual_symptoms[5:12]  # Different set
        
        print(f"   Testing with {len(test_symptoms)} symptoms")
        print(f"   Symptoms: {test_symptoms[:3]}...")
        
        predictions, matched = self.predictor.predict(
            test_symptoms,
            return_top_n=3,
            confidence_threshold=0.1  # Realistic threshold
        )
        
        print(f"   Matched: {len(matched)} symptoms")
        print(f"   Predictions: {len(predictions)}")
        
        if predictions:
            print("\n   Top 3 predictions:")
            for i, (disease, prob, _) in enumerate(predictions, 1):
                print(f"   {i}. {disease}: {prob:.2%}")
        
        self.assertGreater(len(predictions), 0, "Should predict with realistic input")
        print("   ✓ Test passed\n")

if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING DISEASE PREDICTOR TESTS")
    print("="*60)
    unittest.main(verbosity=2)