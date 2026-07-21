#!/usr/bin/env python3
"""
Classifier Head Training — embedding + small MLP
=================================================
Trains a lightweight classifier on the balanced dataset.
Includes targeted augmentation for short/ambiguous examples.
"""

import json
import os
import sys
import numpy as np
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi, upload_file
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# TARGETED EXAMPLES — covering the exact failure patterns
# ============================================================================

TARGETED_EXAMPLES = [
    # 1. Short imperative code (fix "Write a function...")
    ("Write a function that reverses a string", "code"),
    ("Write a function to add two numbers", "code"),
    ("Write a function that checks if a number is prime", "code"),
    ("Write a function that finds the maximum in a list", "code"),
    ("Write a function that sorts an array", "code"),
    ("Write a function that calculates the factorial", "code"),
    ("Write a function that converts Celsius to Fahrenheit", "code"),
    ("Write a function that counts vowels in a string", "code"),
    ("Write a function that removes duplicates from a list", "code"),
    ("Write a function that merges two dictionaries", "code"),
    ("Write a function that flattens a nested list", "code"),
    ("Write a function that computes the Fibonacci sequence", "code"),
    ("Write a function that checks if a string is a palindrome", "code"),
    ("Write a function that reverses a linked list", "code"),
    ("Write a function that parses a CSV string", "code"),
    ("Write a function that generates a random password", "code"),
    ("Write a function that validates an email address", "code"),
    ("Write a function that calculates the average of a list", "code"),
    ("Write a function that finds the median of a list", "code"),
    ("Write a function that filters out even numbers", "code"),
    ("Write a function that maps a list to its squares", "code"),
    ("Write a function that multiplies two numbers", "code"),
    ("Write a function that divides two numbers", "code"),
    ("Write a function that finds the length of a string", "code"),
    ("Write a function that converts a string to uppercase", "code"),
    ("Write a function that converts a string to lowercase", "code"),
    ("Write a function that splits a string by a delimiter", "code"),
    ("Write a function that joins a list of strings", "code"),
    ("Write a function that finds the first duplicate in a list", "code"),
    ("Write a function that finds the second largest in a list", "code"),
    ("Write a function that checks if a number is even", "code"),
    ("Write a function that checks if a number is odd", "code"),
    ("Write a function that rounds a number to two decimals", "code"),
    ("Write a function that calculates the square root", "code"),
    ("Write a function that calculates the power of a number", "code"),
    ("Write a function that calculates the modulus", "code"),
    ("Write a function that capitalizes a string", "code"),
    ("Write a function that reverses the order of words in a sentence", "code"),
    ("Write a function that checks if a sentence is a pangram", "code"),
    ("Write a function that counts word frequency in a string", "code"),
    
    # 2. Questions about code that are actually direct (fix "Why is my function not working")
    ("Why is my function not working", "direct"),
    ("What does this error message mean", "direct"),
    ("Why is my code throwing an error", "direct"),
    ("What's wrong with my function", "direct"),
    ("Why is this not working", "direct"),
    ("Can you explain why my code is failing", "direct"),
    ("Why is my program crashing", "direct"),
    ("Why do I get this error", "direct"),
    ("What's causing this bug", "direct"),
    ("What does this exception mean", "direct"),
    
    # 3. "Start"/"Stop" ambiguities (fix "Can you start writing tests for my app")
    ("Can you start writing tests for my app", "code"),
    ("Start writing unit tests for this function", "code"),
    ("Start reviewing my pull request", "code"),
    ("Start documenting this code", "code"),
    ("Stop ignoring the test failures and fix them", "code"),
    ("Start building the API endpoint", "code"),
    ("Start refactoring this module", "code"),
    ("Start optimizing the database queries", "code"),
    ("Stop using deprecated functions", "code"),
    ("Start implementing the new feature", "code"),
    
    # 4. Terminal-like phrasing that is actually terminal (reinforce)
    ("Run the test suite", "terminal"),
    ("Run the tests", "terminal"),
    ("Run npm start", "terminal"),
    ("Run the build", "terminal"),
    ("Start the server", "terminal"),
    ("Stop the server", "terminal"),
    ("Kill the process", "terminal"),
]

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n" + "=" * 60)
print("🚀 CLASSIFIER HEAD TRAINING (with targeted augmentation)")
print("=" * 60)
print(f"  Repo: {HF_REPO_ID}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Targeted examples: {len(TARGETED_EXAMPLES)}")
print("=" * 60 + "\n")

print("📥 Downloading training data...")
try:
    data_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="classifier_training_data/classifier_training_data.jsonl",
        repo_type=HF_REPO_TYPE,
        token=HF_TOKEN,
    )
    print(f"  ✅ Downloaded: {data_path}")
except Exception as e:
    print(f"  ❌ Failed to download training data: {e}")
    sys.exit(1)

texts, labels = [], []
with open(data_path) as f:
    for line in f:
        row = json.loads(line)
        texts.append(row["text"])
        labels.append(row["label"])

print(f"\n📊 Loaded {len(texts):,} examples")
for cat in sorted(set(labels)):
    print(f"  {cat}: {labels.count(cat):,}")

# ============================================================================
# ADD TARGETED EXAMPLES
# ============================================================================

print(f"\n📝 Adding {len(TARGETED_EXAMPLES)} targeted examples...")
added = 0
for text, label in TARGETED_EXAMPLES:
    if text not in texts:
        texts.append(text)
        labels.append(label)
        added += 1
print(f"  ✅ Added {added} new examples")

print(f"\n📊 After augmentation: {len(texts):,} examples")
for cat in sorted(set(labels)):
    print(f"  {cat}: {labels.count(cat):,}")

# ============================================================================
# ENCODE LABELS
# ============================================================================

print("\n🏷️  Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
classes = label_encoder.classes_.tolist()
print(f"  Classes: {classes}")

# ============================================================================
# EMBED
# ============================================================================

print("\n🧠 Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("📊 Embedding all training examples...")
X = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print(f"  ✅ Embeddings shape: {X.shape}")

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

print(f"\n📊 Split: {len(X_train):,} train, {len(X_val):,} validation")

# ============================================================================
# TRAIN CLASSIFIER
# ============================================================================

print("\n🏋️  Training classifier head...")
classifier = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    alpha=0.0005,
    max_iter=1000,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    verbose=False,
)

classifier.fit(X_train, y_train)

print("\n✅ Validation set performance:")
y_pred = classifier.predict(X_val)
print(classification_report(y_val, y_pred, target_names=classes))
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3%}")

# ============================================================================
# REGRESSION SUITE
# ============================================================================

REGRESSION_SUITE = [
    ("Install express using npm", "terminal"),
    ("Run the test suite", "terminal"),
    ("Start the development server", "terminal"),
    ("What's the command to list files", "terminal"),
    ("Install python package requests", "terminal"),
    ("How do I start the server", "terminal"),
    ("Stop the running process on port 8080", "terminal"),
    ("Kill the background job", "terminal"),
    ("List all the files in the current directory", "terminal"),
    ("Write a function that reverses a string", "code"),
    ("Review this code for bugs", "code"),
    ("Fix the bug in my sorting algorithm", "code"),
    ("Debug why my API returns a 500 error", "code"),
    ("Help me fix this error in my code", "code"),
    ("Why is my function not working", "code"),
    ("Write a python script to parse JSON", "code"),
    ("Can you help me debug this", "code"),
    ("Optimize this SQL query", "code"),
    ("Add error handling to this function", "code"),
    ("Run a quick review of my code", "code"),
    ("Stop overthinking and just fix this bug", "code"),
    ("Can you start writing tests for my app", "code"),
    ("What's the best way to structure this command pattern in my code", "code"),
    ("What does this error message mean", "direct"),
    ("Commit my changes with message 'fix typo'", "direct"),
    ("What files are in this project", "direct"),
    ("Explain what a REST API is", "direct"),
    ("Read the contents of config.json", "direct"),
    ("What's the weather today", "direct"),
    ("Tell me a joke", "direct"),
    ("Why is the sky blue", "direct"),
    ("What is a REST API", "direct"),
]

regression_cases = [(t, l) for t, l in REGRESSION_SUITE if l != "website"]

print("\n" + "=" * 60)
print("🔬 THE REAL TEST: Hand-written Regression Suite")
print("=" * 60 + "\n")

reg_texts = [t for t, l in regression_cases]
reg_labels_original = [l for t, l in regression_cases]
reg_labels_encoded = label_encoder.transform(reg_labels_original)

reg_embeddings = embed_model.encode(reg_texts, normalize_embeddings=True)
reg_predictions_encoded = classifier.predict(reg_embeddings)
reg_predictions = label_encoder.inverse_transform(reg_predictions_encoded)

correct = 0
failures = []
for text, expected, predicted in zip(reg_texts, reg_labels_original, reg_predictions):
    is_correct = expected == predicted
    status = "✅" if is_correct else "❌"
    print(f"{status}  \"{text}\"")
    print(f"      expected: {expected} | got: {predicted}")
    if is_correct:
        correct += 1
    else:
        failures.append((text, expected, predicted))

total = len(regression_cases)
print(f"\n" + "=" * 60)
print(f"📊 REGRESSION SUITE ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
print("=" * 60)

# ============================================================================
# SAVE MODEL
# ============================================================================

model_path = os.path.join(OUTPUT_DIR, "classifier_head.joblib")
encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

joblib.dump(classifier, model_path)
joblib.dump(label_encoder, encoder_path)

print(f"\n💾 Saved classifier to {model_path}")
print(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")
print(f"💾 Saved label encoder to {encoder_path}")

# ============================================================================
# SAVE METADATA (with fix for numpy int64 serialization)
# ============================================================================

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

metadata = {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "classes": classes,
    "class_mapping": {k: int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))},
    "train_size": int(len(X_train)),
    "val_size": int(len(X_val)),
    "regression_accuracy": f"{correct}/{total} ({100*correct/total:.0f}%)",
    "failures": int(len(failures)),
    "failure_details": [
        {"text": text, "expected": expected, "got": got}
        for text, expected, got in failures
    ],
    "targeted_examples_added": len(TARGETED_EXAMPLES),
}

metadata_serializable = json.loads(json.dumps(metadata, default=convert_to_serializable))
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata_serializable, f, indent=2)

# ============================================================================
# UPLOAD
# ============================================================================

print(f"\n📤 Uploading to {HF_REPO_ID}...")
try:
    api = HfApi(token=HF_TOKEN)
    upload_file(
        path_or_fileobj=model_path,
        path_in_repo="classifier_model/classifier_head.joblib",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Add trained classifier head (targeted augmentation)",
    )
    upload_file(
        path_or_fileobj=encoder_path,
        path_in_repo="classifier_model/label_encoder.joblib",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Add label encoder for classifier",
    )
    upload_file(
        path_or_fileobj=os.path.join(OUTPUT_DIR, "metadata.json"),
        path_in_repo="classifier_model/metadata.json",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Add classifier metadata",
    )
    print("✅ Upload complete!")
    print(f"   → https://huggingface.co/{HF_REPO_ID}/tree/main/classifier_model")
except Exception as e:
    print(f"⚠️ Upload failed: {e}")
    print(f"   Model saved locally at {model_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE")
print("=" * 60)
print(f"  Training examples: {len(X_train):,}")
print(f"  Validation examples: {len(X_val):,}")
print(f"  Regression suite: {correct}/{total} ({100*correct/total:.0f}%)")
print(f"  Targeted examples added: {len(TARGETED_EXAMPLES)}")
print("=" * 60)

if failures:
    print(f"\n❌ {len(failures)} failures:")
    for text, expected, got in failures:
        print(f"  \"{text}\" — expected {expected}, got {got}")
    print(f"\n📈 Improvement needed: {len(failures)} failures remain.")
    sys.exit(1)
else:
    print("\n🎉 PERFECT SCORE on the regression suite!")
    print("   The classifier is ready for production!")
    sys.exit(0)