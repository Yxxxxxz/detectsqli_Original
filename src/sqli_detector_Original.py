import re
import urllib.parse
import numpy as np
import pandas as pd
import joblib

from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from difflib import SequenceMatcher


# ======================================================
# Hybrid SQLi Detector
# ======================================================

class SQLiDetector:

    def __init__(self):

        self.w2v_model = None
        self.scaler = None
        self.rf_model = None

        # Signature patterns
        self.signature_patterns = {

            "union-based": [
                r"union\s+select",
                r"union\s+all\s+select"
            ],

            "error-based": [
                r"extractvalue\s*\(",
                r"updatexml\s*\(",
                r"floor\s*\(\s*rand\s*\(",
                r"group\s+by\s+.*rand",
                r"information_schema",
                r"mysql_fetch",
                r"syntax\s+error"
            ],

            "time-based": [
                r"sleep\s*\(",
                r"benchmark\s*\(",
                r"pg_sleep\s*\(",
                r"waitfor\s+delay"
            ]
        }

    # ======================================================
    # Data Cleaning
    # ======================================================

    def clean_data(self, df):

        print("Cleaning dataset...")

        df = df.dropna(subset=["payload"])
        df = df.drop_duplicates()

        df["payload"] = df["payload"].astype(str)

        df = df[df["payload"].str.strip() != ""]
        df = df[df["payload"].str.len() > 3]

        return df

    # ======================================================
    # Normalization
    # ======================================================

    def normalize(self, text):

        text = str(text)

        text = urllib.parse.unquote(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    # ======================================================
    # Tokenization
    # ======================================================

    def tokenize_sql(self, query):

        tokens = re.findall(
            r"[a-zA-Z_]+|\d+|!=|==|<=|>=|['\"].*?['\"]|[^\s]",
            str(query)
        )

        return tokens

    # ======================================================
    # Word2Vec Vector
    # ======================================================

    def get_vector(self, tokens):

        # 1️⃣ ดึงเวกเตอร์ของคำที่ Word2Vec รู้จัก
        vectors = [
            self.w2v_model.wv[word]
            for word in tokens
            if word in self.w2v_model.wv
        ]

        # 2️⃣ ถ้าไม่มีคำที่รู้จักเลย
        if len(vectors) == 0:

            # ส่ง vector ศูนย์แทน (ป้องกัน np.mean error)
            return np.zeros(self.w2v_model.vector_size)

        # 3️⃣ ถ้ามีคำที่รู้จัก → ใช้ค่าเฉลี่ยของ vector
        return np.mean(vectors, axis=0)

    # ======================================================
    # Fuzzy Similarity
    # ======================================================

    def fuzzy_similarity(self, a, b):

        return SequenceMatcher(None, a, b).ratio()

    # ======================================================
    # Signature Detection
    # ======================================================

    def signature_check(self, payload):

        payload = self.normalize(payload)

        for category, patterns in self.signature_patterns.items():

            for pattern in patterns:

                # Exact match
                if re.search(pattern, payload):
                    return True, f"{category} signature"

                # Fuzzy match
                score = self.fuzzy_similarity(payload, pattern)

                if score > 0.75:
                    return True, f"fuzzy {category} signature"

        return False, None

    # ======================================================
    # Train + Evaluate
    # ======================================================

    def train_and_evaluate(self, df):

        df = self.clean_data(df)

        df["payload"] = df["payload"].apply(self.normalize)

        X = df["payload"]
        y = df["label"].values

        print("\nLabel Distribution")
        print(df["label"].value_counts())
        print("Total records:", len(df))

        # Train/Test split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.1,
            stratify=y,
            random_state=42
        )

        # Tokenization
        X_train_tokens = X_train_raw.apply(self.tokenize_sql)
        X_test_tokens = X_test_raw.apply(self.tokenize_sql)

        # Train Word2Vec
        print("\nTraining Word2Vec...")

        self.w2v_model = Word2Vec(
            sentences=X_train_tokens,
            vector_size=150,
            window=5,
            min_count=2,
            workers=4,
            sg=1,
            epochs=20
        )

        # Vectorization
        X_train_vec = np.vstack(X_train_tokens.apply(self.get_vector))
        X_test_vec = np.vstack(X_test_tokens.apply(self.get_vector))

        # Scaling
        self.scaler = StandardScaler()

        X_train_scaled = self.scaler.fit_transform(X_train_vec)
        X_test_scaled = self.scaler.transform(X_test_vec)

        # Train RandomForest
        print("\nTraining RandomForest...")

        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        self.rf_model.fit(X_train_scaled, y_train)

        # Evaluation
        self.evaluate(X_test_scaled, y_test)

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(self, X_test, y_test):

        print("\n==============================")
        print("Performance Results")
        print("==============================")

        y_pred = self.rf_model.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))

        print("\nClassification Report")
        print(classification_report(y_test, y_pred))

        print("\nConfusion Matrix")
        print(confusion_matrix(y_test, y_pred))

    # ======================================================
    # Save Model (Single PKL)
    # ======================================================

    def save_model(self, path="sqli_detector.pkl"):

        joblib.dump(self, path)

        print(f"\nModel saved to {path}")

    # ======================================================
    # Load Model
    # ======================================================

    @staticmethod
    def load_model(path="sqli_detector.pkl"):

        model = joblib.load(path)

        print(f"Model loaded from {path}")

        return model

    # ======================================================
    # Predict Payload
    # ======================================================

    def predict_single(self, payload):

        if self.rf_model is None:
            raise Exception("Model not loaded")

        payload = self.normalize(payload)

        # Stage 1: Signature
        sig_detect, reason = self.signature_check(payload)

        if sig_detect:

            return {
                "prediction": "BLOCKED",
                "stage": "Signature Detection",
                "reason": reason
            }

        # Stage 2: ML
        tokens = self.tokenize_sql(payload)

        vec = self.get_vector(tokens)

        vec_scaled = self.scaler.transform([vec])

        pred = self.rf_model.predict(vec_scaled)[0]

        prob = self.rf_model.predict_proba(vec_scaled)[0][1]

        return {
            "prediction": "BLOCKED" if pred == 1 else "ALLOW",
            "stage": "RandomForest ML",
            "malicious_probability": float(prob)
        }

