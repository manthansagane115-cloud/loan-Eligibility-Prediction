# app.py
import streamlit as st
from loan_rules import loan_fields
from model_handler import check_loan_eligibility

st.set_page_config(page_title="Loan Eligibility Predictor", page_icon="💰")
st.title("💰 Loan Eligibility Predictor")

loan_type = st.selectbox(
    "Select Loan Type",
    options=list(loan_fields.keys()),
    format_func=lambda x: x.replace("_", " ").title(),
)

fields = loan_fields[loan_type]
st.subheader(f"{loan_type.title()} Loan Details")

with st.form(key=f"form_{loan_type}"):
    user_data = {}
    for field in fields:
        name = field["name"]
        label = field["label"]
        field_type = field["type"]

        if field_type == "select":
            user_data[name] = st.selectbox(label, field["choices"], key=f"{loan_type}_{name}")
        elif field_type == "number":
            user_data[name] = st.number_input(label, min_value=0.0, step=1.0, key=f"{loan_type}_{name}")
        else:  # text
            user_data[name] = st.text_input(label, key=f"{loan_type}_{name}")

    submitted = st.form_submit_button("Check Eligibility")

if submitted:
    with st.spinner("Evaluating your application..."):
        try:
            result = check_loan_eligibility(loan_type, user_data)
        except Exception as e:
            st.error(f"Something went wrong while evaluating eligibility: {e}")
        else:
            st.divider()
            st.subheader("Result")

            badge = {
                "Guaranteed Approval": "✅",
                "Likely Approved": "🟢",
                "Needs Improvement": "🟡",
            }.get(result.result, "")

            st.markdown(f"### {badge} {result.result}")

            if result.feedback:
                st.markdown("**Feedback:**")
                for item in result.feedback:
                    st.write(f"- {item}")

            with st.expander("Submitted details"):
                st.json(user_data)
