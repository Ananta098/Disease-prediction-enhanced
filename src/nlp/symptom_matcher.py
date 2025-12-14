import numpy as np
from fuzzywuzzy import fuzz, process
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SymptomMatcher:
    """
    Intelligent symptom matching using multiple techniques:
    1. Exact matching
    2. Fuzzy string matching
    3. Semantic similarity (using word embeddings)
    """
    
    def __init__(self, symptom_vocabulary):
        self.symptom_vocabulary = list(symptom_vocabulary)
        self.nlp = None
        try:
            self.nlp = spacy.load('en_core_web_md')
        except:
            print("Warning: spacy model not loaded. Install with: python -m spacy download en_core_web_md")
        
        # Create TF-IDF vectorizer for symptoms
        self.tfidf = TfidfVectorizer()
        self.symptom_vectors = self.tfidf.fit_transform(self.symptom_vocabulary)
    
    def match_symptom(self, user_input, threshold=80):
        """
        Match user input to closest symptom in vocabulary
        Returns: (matched_symptom, confidence_score)
        """
        user_input = user_input.lower().strip()
        
        # Method 1: Exact match
        if user_input in self.symptom_vocabulary:
            return user_input, 100
        
        # Method 2: Fuzzy string matching
        best_match = process.extractOne(
            user_input, 
            self.symptom_vocabulary,
            scorer=fuzz.token_sort_ratio
        )
        
        if best_match and best_match[1] >= threshold:
            return best_match[0], best_match[1]
        
        # Method 3: Semantic similarity (if spacy is available)
        if self.nlp:
            user_doc = self.nlp(user_input)
            best_similarity = 0
            best_symptom = None
            
            for symptom in self.symptom_vocabulary:
                symptom_doc = self.nlp(symptom)
                similarity = user_doc.similarity(symptom_doc) * 100
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_symptom = symptom
            
            if best_similarity >= threshold * 0.8:  # Lower threshold for semantic
                return best_symptom, best_similarity
        
        # Method 4: TF-IDF cosine similarity
        user_vector = self.tfidf.transform([user_input])
        similarities = cosine_similarity(user_vector, self.symptom_vectors)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx] * 100
        
        if best_score >= threshold * 0.7:
            return self.symptom_vocabulary[best_idx], best_score
        
        return None, 0
    
    def match_multiple_symptoms(self, user_inputs, threshold=80):
        """
        Match multiple user inputs to symptoms
        Returns: list of (matched_symptom, confidence) tuples
        """
        matches = []
        for user_input in user_inputs:
            match, score = self.match_symptom(user_input, threshold)
            if match:
                matches.append((match, score))
        return matches
    
    def suggest_symptoms(self, partial_input, top_n=5):
        """
        Suggest symptoms based on partial input (for autocomplete)
        """
        if not partial_input:
            return []
        
        partial_input = partial_input.lower().strip()
        suggestions = process.extract(
            partial_input,
            self.symptom_vocabulary,
            scorer=fuzz.partial_ratio,
            limit=top_n
        )
        
        return [s[0] for s in suggestions if s[1] > 60]