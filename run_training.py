import os
import json
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

from src.preprocess import clean_text


TRAIN_DIR = "data/raw/train"
TEST_DIR = "data/raw/test"


def load_data(root_dir):
    questions = []
    labels = []

    # Each subfolder = one class
    for label in os.listdir(root_dir):
        label_path = os.path.join(root_dir, label)

        if not os.path.isdir(label_path):
            continue

        for file in os.listdir(label_path):
            if file.endswith(".json"):
                file_path = os.path.join(label_path, file)

                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    question = data.get("problem", "").strip()
                    if question == "":
                        continue

                    questions.append(question)
                    labels.append(label)   # 👈 LABEL FROM FOLDER NAME

    return pd.DataFrame({
        "question": questions,
        "label": labels
    })


# -------- LOAD DATA --------
train_df = load_data(TRAIN_DIR)
test_df = load_data(TEST_DIR)

print("Train samples:", len(train_df))
print("Test samples:", len(test_df))
print("\nTrain label distribution:")
print(train_df["label"].value_counts())

# -------- PREPROCESS --------
X_train = train_df["question"].apply(clean_text)
y_train = train_df["label"]

X_test = test_df["question"].apply(clean_text)
y_test = test_df["label"]

if len(X_train) == 0 or len(X_test) == 0:
    raise ValueError("❌ No data loaded. Check folder structure.")

print("\nSample cleaned question:")
print(X_train.iloc[0])

# -------- TF-IDF --------
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    token_pattern=r"(?u)\b\w+\b"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------- MODEL --------
model = LinearSVC()
model.fit(X_train_vec, y_train)

# -------- EVALUATION --------
y_pred = model.predict(X_test_vec)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))
