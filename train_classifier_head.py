#!/usr/bin/env python3
"""
Classifier Head Training — embedding + small MLP, not a from-scratch model
==============================================================================

Trains a lightweight classifier on top of frozen all-MiniLM-L6-v2 embeddings
to distinguish terminal/code/direct requests. This REUSES the embedding
model already proven for template retrieval — no new model to host/quantize.

Validates against:
  1. A held-out split of the synthetic training data (sanity check)
  2. The 37-case hand-written regression suite (the REAL test — this data
     was never part of generation, so it tests genuine generalization)

GitHub Actions optimized — reads HF_TOKEN and HF_REPO_ID from environment.
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import hf_hub_download, HfApi
from sentence_transformers import SentenceTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG — read from environment
# ============================================================================

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "MR-CODESPIKE/template-library")
HF_REPO_TYPE = "dataset"

if not HF_TOKEN:
    print("❌ HF_TOKEN environment variable not set.")
    sys.exit(1)

# Use GitHub workspace or current directory
BASE_DIR = os.environ.get("GITHUB_WORKSPACE", os.getcwd())
OUTPUT_DIR = os.path.join(BASE_DIR, "classifier_model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: Download training data
# ============================================================================

print("\n" + "=" * 60)
print("🚀 CLASSIFIER HEAD TRAINING")
print("=" * 60)
print(f"  Repo: {HF_REPO_ID}")
print(f"  Output: {OUTPUT_DIR}")
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

print("\n📝 Sample examples per category:")
for cat in ["terminal", "code", "direct"]:
    samples = [t for t, l in zip(texts, labels) if l == cat][:3]
    print(f"\n  {cat}:")
    for s in samples:
        print(f"    - {s}")

# ============================================================================
# STEP 2: Encode labels
# ============================================================================

print("\n🏷️  Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)
print(f"  Classes: {label_encoder.classes_.tolist()}")
print(f"  Encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

# ============================================================================
# STEP 3: Embed all examples
# ============================================================================

print("\n🧠 Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("📊 Embedding all training examples...")
X = embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
print(f"  ✅ Embeddings shape: {X.shape}")

# ============================================================================
# STEP 4: Train/val split
# ============================================================================

X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

print(f"\n📊 Split: {len(X_train):,} train, {len(X_val):,} validation")

# ============================================================================
# STEP 5: Train classifier head
# ============================================================================

print("\n🏋️  Training classifier head...")
classifier = MLPClassifier(
    hidden_layer_sizes=(64, 32),  # Two hidden layers - tiny!
    activation="relu",
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    verbose=False,
)

classifier.fit(X_train, y_train)

print("\n✅ Validation set performance:")
y_pred = classifier.predict(X_val)
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
print(f"Accuracy: {accuracy_score(y_val, y_pred):.3%}")

# ============================================================================
# STEP 6: THE REAL TEST — regression suite
# ============================================================================

REGRESSION_SUITE = [
    # TERMINAL
    ("Install express using npm", "terminal"),
    ("Run the test suite", "terminal"),
    ("Start the development server", "terminal"),
    ("What's the command to list files", "terminal"),
    ("Install python package requests", "terminal"),
    ("How do I start the server", "terminal"),
    ("Stop the running process on port 8080", "terminal"),
    ("Kill the background job", "terminal"),
    ("List all the files in the current directory", "terminal"),
    
    # CODE
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
    
    # DIRECT
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

# Exclude website cases — this classifier only handles terminal/code/direct
regression_cases = [(t, l) for t, l in REGRESSION_SUITE if l != "website"]

print("\n" + "=" * 60)
print("🔬 THE REAL TEST: Hand-written Regression Suite")
print("=" * 60)
print(f"  {len(regression_cases)} cases (NEVER seen during training)")
print("  These test genuine generalization, not memorization")
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

if failures:
    print(f"\n❌ {len(failures)} failure(s):")
    for text, expected, got in failures:
        print(f"  \"{text}\" — expected {expected}, got {got}")
    
    # Show breakdown by expected category
    print("\n📊 Failure breakdown:")
    for cat in ["terminal", "code", "direct"]:
        cat_failures = [f for f in failures if f[1] == cat]
        if cat_failures:
            print(f"  {cat}: {len(cat_failures)} failures")
            for text, expected, got in cat_failures:
                print(f"    - \"{text}\" → got {got}")

# ============================================================================
# STEP 7: Save the classifier and encoder
# ============================================================================

# Save both the classifier and the label encoder
model_path = os.path.join(OUTPUT_DIR, "classifier_head.joblib")
encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")

joblib.dump(classifier, model_path)
joblib.dump(label_encoder, encoder_path)

print(f"\n💾 Saved classifier to {model_path}")
print(f"   Size: {os.path.getsize(model_path) / 1024:.1f} KB")
print(f"💾 Saved label encoder to {encoder_path}")

# Also save the embedding model info
metadata = {
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "classes": label_encoder.classes_.tolist(),
    "class_mapping": dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))),
    "train_size": len(X_train),
    "val_size": len(X_val),
    "regression_accuracy": f"{correct}/{total} ({100*correct/total:.0f}%)",
    "failures": len(failures),
}
with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ============================================================================
# STEP 8: Upload to Hugging Face
# ============================================================================

print(f"\n📤 Uploading to {HF_REPO_ID}...")
try:
    api = HfApi(token=HF_TOKEN)
    
    # Upload classifier
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="classifier_model/classifier_head.joblib",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Add trained classifier head (terminal/code/direct)",
    )
    
    # Upload label encoder
    api.upload_file(
        path_or_fileobj=encoder_path,
        path_in_repo="classifier_model/label_encoder.joblib",
        repo_id=HF_REPO_ID,
        repo_type=HF_REPO_TYPE,
        commit_message="Add label encoder for classifier",
    )
    
    # Upload metadata
    api.upload_file(
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
print(f"  Model size: {os.path.getsize(model_path) / 1024:.1f} KB")
print("=" * 60)

if failures:
    print("\n⚠️  There are failures in the regression suite.")
    print("   Consider collecting more training data for the misclassified cases.")
    sys.exit(1)
else:
    print("\n🎉 PERFECT SCORE on the regression suite!")
    print("   The classifier is ready for production!")
    sys.exit(0)