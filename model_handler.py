import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

DEFAULT_MODEL_PATH = os.path.join("models", "rf_model.joblib")
RANDOM_STATE = 42

class RandomForestModelHandler:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, random_state: int = RANDOM_STATE):
        self.model_path = model_path
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.best_params_: Optional[Dict[str, Any]] = None
        self._build_pipeline()

    def _build_pipeline(self) -> None:
        numeric_features = ["age", "annual_income", "loan_amount", "existing_emis"]
        categorical_features = ["employment_type", "residence_type", "credit_history"]
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ], remainder="drop")
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])

    def prepare_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        required_cols = ["age", "annual_income", "loan_amount", "existing_emis",
                         "employment_type", "residence_type", "credit_history", "approved"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"missing required columns: {missing}")
        X = df[["age", "annual_income", "loan_amount", "existing_emis",
                "employment_type", "residence_type", "credit_history"]].copy()
        y = df["approved"].astype(int).copy()
        return X, y

    def split(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple:
        return train_test_split(X, y, test_size=test_size, random_state=self.random_state, stratify=y)

    def hyperparameter_search(self, X: pd.DataFrame, y: pd.Series, cv_splits: int = 5) -> None:
        param_grid = {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__min_samples_leaf": [1, 2]
        }
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)
        search = GridSearchCV(self.pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        search.fit(X, y)
        self.pipeline = search.best_estimator_
        self.best_params_ = search.best_params_

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        raise RuntimeError("Training disabled in placeholder module")

    def fit_quick(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.pipeline is None:
            self._build_pipeline()
        self.pipeline.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("model pipeline not initialized")
        preds = self.pipeline.predict(X)
        probs = None
        try:
            probs = self.pipeline.predict_proba(X)[:, 1]
        except Exception:
            probs = np.zeros(len(preds))
        acc = accuracy_score(y, preds)
        auc = roc_auc_score(y, probs) if probs is not None else 0.0
        report = classification_report(y, preds, output_dict=True)
        return {"accuracy": acc, "roc_auc": auc, "report": report}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model pipeline not initialized")
        return self.pipeline.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("model pipeline not initialized")
        return self.pipeline.predict(X)

    def save(self, path: Optional[str] = None) -> None:
        target = path or self.model_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "best_params": self.best_params_}, target)

    def load(self, path: Optional[str] = None) -> None:
        target = path or self.model_path
        if not os.path.exists(target):
            raise FileNotFoundError(f"model file not found: {target}")
        data = joblib.load(target)
        self.pipeline = data.get("pipeline")
        self.best_params_ = data.get("best_params")


from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    import streamlit as st
except ImportError:
    st = None


def _get_google_api_key() -> str:
    """
    Resolve the Gemini API key without hardcoding it in source.

    Order of precedence:
      1. Streamlit secrets (st.secrets["GOOGLE_API_KEY"]) — set this in
         Streamlit Cloud under App settings -> Secrets.
      2. GOOGLE_API_KEY environment variable — useful for local dev
         or non-Streamlit deployments.
    """
    if st is not None:
        try:
            key = st.secrets.get("GOOGLE_API_KEY")
            if key:
                return key
        except Exception:
            pass  # no secrets.toml configured locally; fall through to env var

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Set it in Streamlit Cloud "
            "(App settings -> Secrets -> GOOGLE_API_KEY = \"...\") "
            "or as an environment variable for local runs."
        )
    return key


GOOGLE_API_KEY = _get_google_api_key()

# Initialize Gemini
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)

# Pydantic schema for structured output
class LoanEvaluation(BaseModel):
    result: Literal["Likely Approved", "Guaranteed Approval", "Needs Improvement"] = Field(
        description="Final loan eligibility evaluation"
    )
    feedback: List[str] = Field(
        default_factory=lambda: ["No specific feedback provided."],
        description="List of improvement suggestions for approval"
    )

# Wrap model with structured output
evaluation_model = model.with_structured_output(LoanEvaluation)

def check_loan_eligibility(loan_type: str, user_data: dict) -> dict:
    # Calculate monthly disposable income
    monthly_income = 0
    if 'annual_income' in user_data:
        monthly_income = float(user_data['annual_income']) / 12
    elif 'monthly_income' in user_data:
        monthly_income = float(user_data['monthly_income'])
    elif 'business_turnover' in user_data:
        monthly_income = (
            float(user_data['business_turnover'])
            * float(user_data['profit_margin']) / 100
            / 12
        )
    elif 'annual_family_income' in user_data:
        monthly_income = float(user_data['annual_family_income']) / 12

    other_expenses = float(user_data.get('other_expenses', 0))
    disposable_income = monthly_income - other_expenses

    prompt = f"""
You are a loan eligibility predictor. Consider the following calculations:

Monthly Income: ₹{monthly_income:,.2f}
Monthly Other Expenses: ₹{other_expenses:,.2f}
Monthly Disposable Income: ₹{disposable_income:,.2f}

Loan Type: {loan_type}
User Data: {user_data}

Based on the monthly disposable income and other factors, evaluate the loan eligibility.
"""

    response = evaluation_model.invoke(prompt)
    return response
