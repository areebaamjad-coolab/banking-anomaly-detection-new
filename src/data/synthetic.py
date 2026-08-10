"""Generate synthetic master CSV (transactions + profile + demographics + features)."""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta

import polars as pl

from src.data.database import LocalDatabase
from src.data.feature_engineering import engineer_features
from src.data.preprocessor import preprocess


CHANNELS = ["ATM", "IB", "BRANCH TELLER", "1LINK", "VISA", "INTERNAL"]
CURRENCIES = ["PKR", "USD"]
TXN_TYPES = ["AC", "DC", "TT", "PO"]
BANKS = ["HBL", "UBL", "MCB", "ABL", "NBP"]


def generate_master_csv(
    n_accounts: int = 50,
    txns_per_account: int = 100,
    start_date: datetime | None = None,
    save: bool = True,
) -> pl.DataFrame:
    random.seed(42)
    start = start_date or datetime(2024, 1, 1)
    rows: list[dict] = []

    for acct_i in range(n_accounts):
        account_id = f"ACC{acct_i:06d}"
        customer_id = f"CUST{acct_i // 3:05d}"
        open_date = start - timedelta(days=random.randint(365, 2000))
        branch = f"PK001{random.randint(1000, 9999):04d}"
        profile = {
            "account_open_date": open_date.strftime("%Y-%m-%d"),
            "account_type": random.choice(["SAVINGS", "CURRENT", "FIXED"]),
            "customer_segment": random.choice(["RETAIL", "SME", "CORPORATE"]),
            "branch_code_profile": branch,
            "city": random.choice(["Karachi", "Lahore", "Islamabad"]),
            "occupation": random.choice(["SALARIED", "BUSINESS", "PROFESSIONAL"]),
            "yearly_income": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "risk_rating": random.choice(["LOW", "MEDIUM", "HIGH"]),
            "account_age_days": (start - open_date).days,
            "account_age_years": round((start - open_date).days / 365.25, 2),
            "new_account_indicator": 1 if (start - open_date).days < 365 else 0,
            "customer_age": random.randint(22, 65),
            "num_accounts_per_customer": random.randint(1, 4),
            "total_customer_bal": round(random.uniform(10_000, 5_000_000), 2),
            "is_corporate": 1 if acct_i % 7 == 0 else 0,
            "is_individual": 1 if acct_i % 7 != 0 else 0,
            "is_dormant_customer_flag": 0,
            "is_business_account": 1 if acct_i % 7 == 0 else 0,
            "is_personal_account": 1 if acct_i % 7 != 0 else 0,
            "customer_type": random.choice(["INDIVIDUAL", "CORPORATE"]),
            "industry_desc": random.choice(["TRADE", "MANUFACTURING", "SERVICES"]),
            "occupation_desc": random.choice(["ENGINEER", "TRADER", "DOCTOR"]),
            "account_status_desc": "ACTIVE",
            "kyc_risk_category_desc_in_review": random.choice(["LOW", "MEDIUM"]),
            "risk_profile_description": random.choice(["STANDARD", "ENHANCED"]),
            "dominant_credit_mode": random.choice(CHANNELS),
            "dominant_debit_mode": random.choice(CHANNELS),
        }
        balance = random.uniform(50_000, 500_000)

        for t in range(txns_per_account):
            ts = start + timedelta(
                days=random.randint(0, 400),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59),
            )
            is_debit = random.random() > 0.45
            amount = round(random.lognormvariate(8, 1.5), 2)
            if is_debit:
                balance_before = balance + amount
                balance_after = balance
                balance = balance_after
            else:
                balance_before = balance
                balance_after = balance + amount
                balance = balance_after

            rows.append({
                "transaction_id": str(uuid.uuid4())[:12].upper(),
                "customer_id": customer_id,
                "account_id": account_id,
                "txn_timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "amount": amount,
                "currency": random.choice(CURRENCIES),
                "transaction_type": random.choice(TXN_TYPES),
                "debit_credit": "debit" if is_debit else "credit",
                "channel": random.choice(CHANNELS),
                "branch_code": branch,
                "beneficiary_account": f"BEN{random.randint(100000,999999)}",
                "beneficiary_bank_issuer": random.choice(BANKS),
                "beneficiary_bank_acquirer": random.choice(BANKS),
                "status": "success",
                "balance_before": round(balance_before, 2),
                "balance_after": round(balance_after, 2),
                "transaction_type_desc": random.choice(["TRANSFER", "CASH WITHDRAWAL", "SALARY"]),
                "credit_acct_no": account_id if not is_debit else f"EXT{random.randint(1000,9999)}",
                "debit_acct_no": account_id if is_debit else f"EXT{random.randint(1000,9999)}",
                **profile,
            })

    df = pl.DataFrame(rows)
    df = engineer_features(df)
    df = preprocess(df)

    if save:
        db = LocalDatabase()
        db.write(df, backup=False)

    return df
