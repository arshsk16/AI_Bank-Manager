import math
import numpy as np
from datetime import datetime

def check_eligibility(loan_data, document_data):
    """
    Check loan eligibility based on provided loan and document data.
    
    Args:
        loan_data (dict): Loan application data
        document_data (dict): Extracted document information
    
    Returns:
        dict: Eligibility result with status and details
    """
    # Extract loan details
    loan_type = loan_data.get("loan_type", "")
    loan_amount = loan_data.get("loan_amount", 0)
    loan_tenure = loan_data.get("loan_tenure", 0)
    monthly_income = loan_data.get("monthly_income", 0)
    existing_loans = loan_data.get("existing_loans", 0)
    employment_status = loan_data.get("employment_status", "")
    
    # Calculate basic eligibility
    result = {}
    
    # Minimum income requirements by loan type
    min_income_requirements = {
        "Personal Loan": 20000,
        "Home Loan": 35000,
        "Education Loan": 25000,
        "Vehicle Loan": 30000,
        "Business Loan": 50000
    }
    
    min_income_required = min_income_requirements.get(loan_type, 25000)
    
    # Maximum eligible loan amount is typically 10-20 times monthly income depending on loan type
    loan_multipliers = {
        "Personal Loan": 10,
        "Home Loan": 60,
        "Education Loan": 15,
        "Vehicle Loan": 20,
        "Business Loan": 30
    }
    
    # Risk factors based on employment status
    risk_factors = {
        "Salaried": 1.0,  # Lowest risk
        "Self-employed": 1.2,
        "Business Owner": 1.3,
        "Retired": 1.5,
        "Unemployed": 2.0  # Highest risk
    }
    
    risk_factor = risk_factors.get(employment_status, 1.5)
    
    # Calculate EMI burden (existing EMIs / monthly income)
    emi_burden_ratio = existing_loans / monthly_income if monthly_income > 0 else 1.0
    
    # Maximum allowed EMI burden
    max_emi_burden = 0.5  # 50% of income
    
    # Check different eligibility factors
    rejection_reasons = []
    required_info = []
    
    # Income check
    if monthly_income < min_income_required:
        rejection_reasons.append(f"Monthly income below minimum requirement of ₹{min_income_required:,} for {loan_type}")
    
    # EMI burden check
    if emi_burden_ratio > max_emi_burden:
        rejection_reasons.append(f"Existing EMI obligations exceed {max_emi_burden*100}% of monthly income")
    
    # Employment status check for certain loan types
    if loan_type in ["Home Loan", "Vehicle Loan"] and employment_status == "Unemployed":
        rejection_reasons.append(f"{employment_status} status not eligible for {loan_type}")
    
    # Document verification checks
    if "aadhaar" not in document_data:
        required_info.append("Aadhaar card verification")
    
    if "pan" not in document_data:
        required_info.append("PAN card verification")
    
    # For higher loan amounts, income proof is mandatory
    if loan_amount > 500000 and "income_proof" not in document_data:
        required_info.append("Income proof for loan amount exceeding ₹500,000")
    
    # Calculate eligible amount based on multiplier and risk factor
    loan_multiplier = loan_multipliers.get(loan_type, 15) / risk_factor
    max_eligible_amount = monthly_income * loan_multiplier * 12  # Annual basis
    
    # Apply EMI burden adjustment
    remaining_income_capacity = monthly_income * (max_emi_burden - emi_burden_ratio)
    max_emi_capacity = remaining_income_capacity * 0.9  # 90% of remaining capacity
    
    # Calculate maximum eligible amount based on EMI capacity
    interest_rate = get_interest_rate(loan_type, loan_tenure, risk_factor)
    max_eligible_by_emi = calculate_loan_from_emi(max_emi_capacity, interest_rate, loan_tenure)
    
    # Take the lower of the two eligible amounts
    max_eligible_amount = min(max_eligible_amount, max_eligible_by_emi)
    
    # Determine final eligibility status
    if len(required_info) > 0:
        # More information needed
        result["status"] = "more_info_needed"
        result["required_info"] = required_info
        
    elif len(rejection_reasons) > 0:
        # Application rejected
        result["status"] = "rejected"
        result["rejection_reasons"] = rejection_reasons
        
    else:
        # Application approved
        approved_amount = min(loan_amount, max_eligible_amount)
        
        # Round down to nearest 10000
        approved_amount = math.floor(approved_amount / 10000) * 10000
        
        result["status"] = "approved"
        result["approved_amount"] = approved_amount
        result["interest_rate"] = interest_rate
        result["tenure"] = loan_tenure
        
        # Calculate EMI
        monthly_emi = calculate_emi(approved_amount, interest_rate, loan_tenure)
        
        result["monthly_emi"] = monthly_emi
        result["total_interest"] = (monthly_emi * loan_tenure * 12) - approved_amount
        result["processing_fee"] = approved_amount * 0.01  # 1% of loan amount
    
    return result


def get_interest_rate(loan_type, loan_tenure, risk_factor):
    """
    Calculate the applicable interest rate based on loan type, tenure and risk factor.
    
    Args:
        loan_type (str): Type of loan
        loan_tenure (int): Loan tenure in years
        risk_factor (float): Risk factor based on employment status
    
    Returns:
        float: Annual interest rate percentage
    """
    # Base interest rates by loan type
    base_rates = {
        "Personal Loan": 12.5,
        "Home Loan": 8.5,
        "Education Loan": 9.5,
        "Vehicle Loan": 10.5,
        "Business Loan": 14.0
    }
    
    base_rate = base_rates.get(loan_type, 12.0)
    
    # Tenure adjustment: longer tenure = slightly higher rate
    tenure_adjustment = 0.0
    if loan_tenure > 5:
        tenure_adjustment = 0.5
    if loan_tenure > 10:
        tenure_adjustment = 1.0
    if loan_tenure > 15:
        tenure_adjustment = 1.5
    
    # Calculate final rate with risk adjustment
    final_rate = base_rate + tenure_adjustment
    final_rate = final_rate * (risk_factor ** 0.5)  # Square root to dampen the effect
    
    # Round to 1 decimal place
    return round(final_rate, 1)


def calculate_emi(principal, interest_rate, tenure_years):
    """
    Calculate Equated Monthly Installment (EMI) for a loan.
    
    Args:
        principal (float): Loan amount
        interest_rate (float): Annual interest rate percentage
        tenure_years (int): Loan tenure in years
    
    Returns:
        float: Monthly EMI amount
    """
    # Convert annual interest rate to monthly rate
    monthly_rate = interest_rate / (12 * 100)
    
    # Convert tenure from years to months
    tenure_months = tenure_years * 12
    
    # Calculate EMI
    emi = (principal * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
    
    return emi


def calculate_loan_from_emi(emi, interest_rate, tenure_years):
    """
    Calculate maximum loan amount based on EMI capacity.
    
    Args:
        emi (float): Monthly EMI amount
        interest_rate (float): Annual interest rate percentage
        tenure_years (int): Loan tenure in years
    
    Returns:
        float: Maximum eligible loan amount
    """
    # Convert annual interest rate to monthly rate
    monthly_rate = interest_rate / (12 * 100)
    
    # Convert tenure from years to months
    tenure_months = tenure_years * 12
    
    # Calculate loan amount
    loan_amount = emi * ((1 + monthly_rate) ** tenure_months - 1) / (monthly_rate * (1 + monthly_rate) ** tenure_months)
    
    return loan_amount