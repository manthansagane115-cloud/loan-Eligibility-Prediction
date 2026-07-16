# app.py
from flask import Flask, render_template, request
from loan_rules import loan_fields
from model_handler import check_loan_eligibility

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/form/<loan_type>")
def form(loan_type):
    if loan_type not in loan_fields:
        return "Invalid Loan Type", 400
    fields = loan_fields[loan_type]
    return render_template("form.html", loan_type=loan_type, fields=fields)

@app.route("/predict/<loan_type>", methods=["POST"])
def predict(loan_type):
    if loan_type not in loan_fields:
        return "Invalid Loan Type", 400
    
    user_data = {field["name"]: request.form.get(field["name"]) for field in loan_fields[loan_type]}
    result = check_loan_eligibility(loan_type, user_data)

    return render_template("result.html", result=result, loan_type=loan_type, user_data=user_data)

if __name__ == "__main__":
    app.run(debug=True)
