
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

##

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df = pd.read_csv(DATA_DIR / "validation.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")

X_train = train_df["text"].fillna("")
y_train = train_df["priority"]

X_val = val_df["text"].fillna("")
y_val = val_df["priority"]

X_test = test_df["text"].fillna("")
y_test = test_df["priority"]

##

def train_model():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight=None,
            random_state=42
        ))
    ])


    model.fit(X_train, y_train)

    return model 

def analyse_model(model):
    print("\nVALIDATION RESULTS")

    val_predictions = model.predict(X_val)

    print(classification_report(
        y_val,
        val_predictions,
        zero_division=0
    ))

    print("\nTEST RESULTS")

    test_predictions = model.predict(X_test)

    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))

    print(model.predict(["the the the the"]))

def save_model(model):
    model_path = BASE_DIR / "ticket_classifier.joblib"
    joblib.dump(model, model_path)

##

if __name__ == "__main__":
    model = train_model()
    analyse_model(model)
    save_model(model)
