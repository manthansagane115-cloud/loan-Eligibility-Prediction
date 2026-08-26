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


def _get_google_api_key() -> Optional[str]:
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
    if key:
        return key
    return None


# Pydantic schema for structured output
class LoanEvaluation(BaseModel):
    result: Literal["Likely Approved", "Guaranteed Approval", "Needs Improvement"] = Field(
        description="Final loan eligibility evaluation"
    )
    feedback: List[str] = Field(
        default_factory=lambda: ["No specific feedback provided."],
        description="List of improvement suggestions for approval"
    )


def _get_evaluation_model():
    key = _get_google_api_key()
    if not key:
        return None
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=key,
            temperature=0.2
        )
        return model.with_structured_output(LoanEvaluation)
    except Exception:
        return None

def _evaluate_loan_rules(loan_type: str, user_data: dict) -> LoanEvaluation:
    """
    Deterministic rule-based fallback logic to evaluate loan eligibility
    when LLM is disabled or unavailable.
    """
    feedback = []

    # Calculate monthly income
    monthly_income = 0.0
    try:
        if 'annual_income' in user_data and user_data['annual_income']:
            monthly_income = float(user_data['annual_income']) / 12
        elif 'monthly_income' in user_data and user_data['monthly_income']:
            monthly_income = float(user_data['monthly_income'])
        elif 'business_turnover' in user_data and user_data['business_turnover']:
            turnover = float(user_data['business_turnover'])
            margin = float(user_data.get('profit_margin', 10) or 10)
            monthly_income = (turnover * margin / 100) / 12
        elif 'annual_family_income' in user_data and user_data['annual_family_income']:
            monthly_income = float(user_data['annual_family_income']) / 12
    except (ValueError, TypeError):
        monthly_income = 0.0

    # Calculate expenses and disposable income
    try:
        other_expenses = float(user_data.get('other_expenses', 0) or 0)
    except (ValueError, TypeError):
        other_expenses = 0.0

    disposable_income = monthly_income - other_expenses

    # Extract loan amount & tenure
    try:
        loan_amount = float(user_data.get('loan_amount', 0) or 0)
    except (ValueError, TypeError):
        loan_amount = 0.0

    try:
        loan_tenure_years = float(user_data.get('loan_tenure', 5) or 5)
        if loan_tenure_years <= 0:
            loan_tenure_years = 5.0
    except (ValueError, TypeError):
        loan_tenure_years = 5.0

    # Credit score check
    has_credit_score = 'credit_score' in user_data and user_data['credit_score'] != ''
    try:
        credit_score = float(user_data.get('credit_score', 700) or 700) if has_credit_score else 700
    except (ValueError, TypeError):
        credit_score = 700
        has_credit_score = False

    # Estimated EMI calculation (approx 10.5% p.a. interest)
    months = loan_tenure_years * 12
    r = 0.105 / 12
    if loan_amount > 0 and months > 0:
        estimated_emi = loan_amount * r * ((1 + r) ** months) / (((1 + r) ** months) - 1)
    else:
        estimated_emi = 0.0

    # Evaluate risk factors
    dti_ratio = (estimated_emi / monthly_income) if monthly_income > 0 else 1.0

    if disposable_income <= 0:
        result = "Needs Improvement"
        feedback.append("Monthly expenses equal or exceed total monthly income.")
        feedback.append("Consider reducing monthly expenses or lowering the requested loan amount.")
    elif dti_ratio <= 0.35 and (credit_score >= 750 or not has_credit_score) and disposable_income >= estimated_emi * 1.5:
        result = "Guaranteed Approval"
        feedback.append("Excellent disposable income relative to requested loan EMI.")
        feedback.append("High financial stability and low debt burden.")
    elif disposable_income >= estimated_emi and (credit_score >= 650 or not has_credit_score) and dti_ratio <= 0.55:
        result = "Likely Approved"
        feedback.append("Sufficient disposable income to cover estimated loan EMIs.")
        feedback.append("Healthy financial profile for loan approval.")
    else:
        result = "Needs Improvement"
        if estimated_emi > disposable_income:
            feedback.append("Estimated monthly EMI exceeds monthly disposable income.")
            feedback.append("Try extending loan tenure or reducing loan amount to lower monthly payments.")
        if has_credit_score and credit_score < 650:
            feedback.append("Credit score is below recommended threshold (650+). Improving your score will increase approval chances.")
        if dti_ratio > 0.5:
            feedback.append("High debt-to-income ratio. Reducing existing debts can boost your eligibility.")

    # Loan type specific recommendations
    if loan_type == "education":
        if user_data.get("co_borrower") == "No":
            feedback.append("Adding a co-borrower can strengthen your application.")
        if user_data.get("guarantor") == "No":
            feedback.append("Providing a guarantor can improve loan terms.")
    elif loan_type in ["home", "car"]:
        try:
            dp = float(user_data.get("down_payment", 0) or 0)
            pv = float(user_data.get("property_value" if loan_type == "home" else "car_price", 0) or 0)
            if pv > 0 and (dp / pv) < 0.15:
                feedback.append("Increasing down payment to at least 15-20% improves approval odds.")
        except (ValueError, TypeError):
            pass

    if not feedback:
        feedback.append("Your application parameters have been evaluated.")

    return LoanEvaluation(result=result, feedback=feedback)


def check_loan_eligibility(loan_type: str, user_data: dict) -> LoanEvaluation:
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

    # Try model evaluation if available
    eval_model = _get_evaluation_model()
    if eval_model is not None:
        try:
            prompt = f"""
You are a loan eligibility predictor. Consider the following calculations:

Monthly Income: ₹{monthly_income:,.2f}
Monthly Other Expenses: ₹{other_expenses:,.2f}
Monthly Disposable Income: ₹{disposable_income:,.2f}

Loan Type: {loan_type}
User Data: {user_data}

Based on the monthly disposable income and other factors, evaluate the loan eligibility.
"""

            response = eval_model.invoke(prompt)
            if response and hasattr(response, 'result') and response.result:
                return response
        except Exception:
            # Fall back to rule-based evaluation if model call fails (e.g. 403 PERMISSION_DENIED, bad API key)
            pass

    return _evaluate_loan_rules(loan_type, user_data)
