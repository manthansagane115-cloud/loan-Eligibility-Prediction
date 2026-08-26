import os
import random
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List, Literal

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

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
            # sparse_output replaces the deprecated/removed `sparse` kwarg (sklearn >= 1.2,
            # `sparse` was removed entirely in 1.4+)
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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
        # Intentionally disabled: use fit_quick() for a fast fit or
        # hyperparameter_search() for a tuned fit. This placeholder exists so
        # any accidental call to .train() fails loudly and immediately rather
        # than silently doing the wrong thing.
        raise RuntimeError(
            "train() is disabled in this module. Use fit_quick() or hyperparameter_search() instead."
        )

    def fit_quick(self, X: pd.DataFrame, y: pd.Series) -> None:
        if self.pipeline is None:
            self._build_pipeline()
        self.pipeline.fit(X, y)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError("model pipeline not initialized")
        preds = self.pipeline.predict(X)
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


# ---------------------------------------------------------------------------
# LLM-based loan eligibility evaluation (Gemini via LangChain)
# ---------------------------------------------------------------------------

# API keys must come from the environment — never hardcode credentials in
# source. Set GOOGLE_API_KEYS as a comma-separated list to enable simple
# round-robin/random key selection (e.g. for spreading across quota), or
# GOOGLE_API_KEY for a single key.
_keys_env = os.environ.get("GOOGLE_API_KEYS")
if _keys_env:
    API_KEYS = [k.strip() for k in _keys_env.split(",") if k.strip()]
else:
    single_key = os.environ.get("GOOGLE_API_KEY")
    API_KEYS = [single_key] if single_key else []

if not API_KEYS:
    raise RuntimeError(
        "No Google API key configured. Set the GOOGLE_API_KEY environment "
        "variable (or GOOGLE_API_KEYS for multiple, comma-separated)."
    )

GOOGLE_API_KEY = random.choice(API_KEYS)

GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")

model = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


class LoanEvaluation(BaseModel):
    result: Literal["Likely Approved", "Needs Improvement", "Unlikely Approved"] = Field(
        description="Final loan eligibility evaluation"
    )
    feedback: List[str] = Field(
        default_factory=lambda: ["No specific feedback provided."],
        description="List of improvement suggestions for approval"
    )


evaluation_model = model.with_structured_output(LoanEvaluation)


def check_loan_eligibility(loan_type: str, user_data: dict) -> dict:
    """
    Evaluate loan eligibility using an LLM given basic financial inputs.

    Raises:
        ValueError: if required income fields are missing/invalid.
        RuntimeError: if the underlying LLM call fails (e.g. API disabled,
            quota exceeded, network error).
    """
    monthly_income = 0.0
    try:
        if 'annual_income' in user_data:
            monthly_income = float(user_data['annual_income']) / 12
        elif 'monthly_income' in user_data:
            monthly_income = float(user_data['monthly_income'])
        elif 'business_turnover' in user_data:
            if 'profit_margin' not in user_data:
                raise ValueError(
                    "business_turnover was provided but profit_margin is missing"
                )
            monthly_income = (
                float(user_data['business_turnover'])
                * float(user_data['profit_margin']) / 100
                / 12
            )
        elif 'annual_family_income' in user_data:
            monthly_income = float(user_data['annual_family_income']) / 12
        else:
            raise ValueError(
                "No recognized income field found in user_data "
                "(expected one of: annual_income, monthly_income, "
                "business_turnover + profit_margin, annual_family_income)"
            )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid income data for eligibility check: {e}") from e

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

    try:
        response = evaluation_model.invoke(prompt)
    except Exception as e:
        raise RuntimeError(f"Loan eligibility evaluation failed: {e}") from e

    return response
