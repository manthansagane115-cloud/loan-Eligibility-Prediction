# loan_rules.py

loan_fields = {
    "home": [
        {"name": "property_value", "label": "Property Value (₹)", "type": "number"},
        {"name": "down_payment", "label": "Down Payment (₹)", "type": "number"},
        {"name": "employment_type", "label": "Employment Type", "type": "select",
         "choices": ["Salaried", "Self-Employed", "Government Job", "Retired"]},
        {"name": "annual_income", "label": "Annual Income (₹)", "type": "number"},
        {"name": "other_expenses", "label": "Monthly Other Expenses (₹)", "type": "number"},
        {"name": "credit_score", "label": "Credit Score", "type": "number"},
        {"name": "existing_loans", "label": "Number of Existing Loans", "type": "number"},
        {"name": "loan_amount", "label": "Loan Amount Requested (₹)", "type": "number"},
        {"name": "loan_tenure", "label": "Loan Tenure (years)", "type": "number"}
    ],

    "education": [
        {"name": "university", "label": "University Name", "type": "text"},
        {"name": "course", "label": "Course Name", "type": "text"},
        {"name": "course_fee", "label": "Course Fee (₹)", "type": "number"},
        {"name": "co_borrower", "label": "Co-borrower Available", "type": "select",
         "choices": ["Yes", "No"]},
        {"name": "guarantor", "label": "Guarantor Available", "type": "select",
         "choices": ["Yes", "No"]},
        {"name": "annual_family_income", "label": "Annual Family Income (₹)", "type": "number"},
        {"name": "other_expenses", "label": "Monthly Other Expenses (₹)", "type": "number"},
        {"name": "credit_score", "label": "Applicant Credit Score", "type": "number"},
        {"name": "loan_amount", "label": "Loan Amount Requested (₹)", "type": "number"}
    ],

    "business": [
        {"name": "business_type", "label": "Business Type", "type": "select",
         "choices": ["Retail", "Manufacturing", "IT Services", "Consulting", "Other"]},
        {"name": "business_turnover", "label": "Annual Business Turnover (₹)", "type": "number"},
        {"name": "years_in_operation", "label": "Years in Operation", "type": "number"},
        {"name": "profit_margin", "label": "Profit Margin (%)", "type": "number"},
        {"name": "existing_loans", "label": "Existing Loans (₹)", "type": "number"},
        {"name": "other_expenses", "label": "Monthly Other Expenses (₹)", "type": "number"},
        {"name": "loan_amount", "label": "Loan Amount Requested (₹)", "type": "number"},
        {"name": "loan_tenure", "label": "Loan Tenure (years)", "type": "number"},
        {"name": "collateral", "label": "Collateral Value (₹)", "type": "number"}
    ],

    "car": [
        {"name": "car_model", "label": "Car Model", "type": "text"},
        {"name": "car_price", "label": "Car Price (₹)", "type": "number"},
        {"name": "down_payment", "label": "Down Payment (₹)", "type": "number"},
        {"name": "employment_type", "label": "Employment Type", "type": "select",
         "choices": ["Salaried", "Self-Employed", "Government Job", "Retired"]},
        {"name": "annual_income", "label": "Annual Income (₹)", "type": "number"},
        {"name": "other_expenses", "label": "Monthly Other Expenses (₹)", "type": "number"},
        {"name": "credit_score", "label": "Credit Score", "type": "number"},
        {"name": "loan_amount", "label": "Loan Amount Requested (₹)", "type": "number"},
        {"name": "loan_tenure", "label": "Loan Tenure (years)", "type": "number"}
    ],

    "personal": [
        {"name": "employment_type", "label": "Employment Type", "type": "select",
         "choices": ["Salaried", "Self-Employed", "Freelancer", "Unemployed"]},
        {"name": "monthly_income", "label": "Monthly Income (₹)", "type": "number"},
        {"name": "other_expenses", "label": "Monthly Other Expenses (₹)", "type": "number"},
        {"name": "credit_score", "label": "Credit Score", "type": "number"},
        {"name": "existing_loans", "label": "Existing Loans (₹)", "type": "number"},
        {"name": "loan_amount", "label": "Loan Amount Requested (₹)", "type": "number"},
        {"name": "loan_purpose", "label": "Purpose of Loan", "type": "select",
         "choices": ["Medical", "Travel", "Education", "Debt Consolidation", "Other"]},
        {"name": "loan_tenure", "label": "Loan Tenure (years)", "type": "number"}
    ]
}
