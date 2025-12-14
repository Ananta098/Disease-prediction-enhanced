import sys
import os
import joblib

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*60)
print("VOCABULARY INSPECTOR")
print("="*60)

# Load vocabulary
vocab_path = 'data/processed/vocabulary.pkl'
vocab_data = joblib.load(vocab_path)

symptoms = vocab_data['symptoms']
label_encoder = vocab_data['label_encoder']

print(f"\nTotal symptoms in vocabulary: {len(symptoms)}")
print(f"Total diseases: {len(label_encoder.classes_)}")

# Show first 50 symptoms
print("\nFirst 50 symptoms:")
print("-" * 60)
for i, symptom in enumerate(sorted(symptoms)[:50], 1):
    print(f"{i:3d}. {symptom}")

# Show all diseases
print("\n\nAll diseases:")
print("-" * 60)
for i, disease in enumerate(label_encoder.classes_, 1):
    print(f"{i:3d}. {disease}")

# Search for specific symptoms
print("\n\nSearching for common symptoms:")
print("-" * 60)
test_symptoms = ['fever', 'cough', 'headache', 'fatigue', 'pain', 'nausea']

for symptom in test_symptoms:
    found = [s for s in symptoms if symptom in s.lower()]
    if found:
        print(f"✓ '{symptom}' found as: {found[:5]}")  # Show first 5 matches
    else:
        print(f"❌ '{symptom}' NOT found")

# Save to file for reference
with open('vocabulary_list.txt', 'w') as f:
    f.write("SYMPTOMS:\n")
    f.write("="*60 + "\n")
    for symptom in sorted(symptoms):
        f.write(f"{symptom}\n")
    
    f.write("\n\nDISEASES:\n")
    f.write("="*60 + "\n")
    for disease in label_encoder.classes_:
        f.write(f"{disease}\n")

print("\n✓ Full vocabulary saved to: vocabulary_list.txt")
print("="*60)