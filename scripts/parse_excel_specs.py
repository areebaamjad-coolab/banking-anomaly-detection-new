"""
Parse both Excel spec files and regenerate config/feature_registry.yaml
and config/feature_definitions.json.

Run from project root (after pip install):
    python scripts/parse_excel_specs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
sys.path.insert(0, str(ROOT))

try:
    import pandas as pd
    import yaml
except ImportError as e:
    print(f"Missing dependency: {e}. Install with: pip install pandas pyyaml openpyxl")
    sys.exit(1)

COLUMN_LIST = PARENT / "coloumn list for model (feature set ready till now ).xlsx"
FEATURE_STORE = PARENT / "Feature Store Audit (1).xlsx"

EXCLUDED_RAW = {"amount", "balance_before", "balance_after"}
EXCLUDED_IDS = {
    "transaction_id", "customer", "customer_id", "account_id", "txn_timestamp",
}


def parse_column_list(path: Path) -> dict:
    df = pd.read_excel(path)
    items = [str(x).strip() for x in df.iloc[:, 0].dropna()]

    ready: list[str] = []
    demo_num, demo_flag, demo_cat = [], [], []
    pending_note: list[str] = []
    mode = "ready"

    for x in items:
        if "not included" in x.lower() or x.startswith("Note:"):
            pending_note.append(x)
            continue
        if x in EXCLUDED_RAW:
            continue
        if "Categorical Columns" in x:
            mode = "cat"
            continue
        if "Coloumns from customer profile" in x:
            mode = "cp"
            continue
        if "demographic_numeric_cols" in x or "Demographic Numeric" in x:
            mode = "demo_num"
            continue
        if "demographic_flag_cols" in x or "Dempgraphic binary" in x:
            mode = "demo_flag"
            continue
        if "demographic_categorical_cols" in x or "Demographic categorical" in x:
            mode = "demo_cat"
            continue
        if x == "]":
            mode = "ready"
            continue
        if "zscore" in x.lower():
            ready.extend(p.strip().strip(",") for p in x.split(",") if p.strip())
            continue
        if mode == "cat" and "currency" in x:
            continue
        if mode == "cp" and x and not x.startswith("Demographic"):
            name = x.split()[0] if " " in x else x
            if re.match(r"^[a-z_]+", name):
                ready.append(name)
            continue
        if mode == "demo_num" and x.startswith('"'):
            demo_num.append(x.strip('",'))
            continue
        if mode == "demo_flag" and x.startswith('"'):
            demo_flag.append(x.strip('",'))
            continue
        if mode == "demo_cat" and x.startswith('"'):
            demo_cat.append(x.strip('",'))
            continue
        if re.match(r"^[a-z][a-z0-9_]*$", x) and mode == "ready":
            ready.append(x)

    ready.extend(demo_num + demo_flag + demo_cat)
    ready = sorted(set(ready))

    return {
        "excluded_from_model": sorted(EXCLUDED_IDS | EXCLUDED_RAW),
        "transaction_categoricals": [
            "currency", "transaction_type", "debit_credit", "channel",
            "beneficiary_bank_issuer", "beneficiary_bank_acquirer",
            "status", "transaction_type_desc",
        ],
        "raw_transaction_columns": [
            "transaction_id", "customer_id", "account_id", "txn_timestamp",
            "amount", "currency", "transaction_type", "debit_credit", "channel",
            "branch_code", "beneficiary_account", "beneficiary_bank_issuer",
            "beneficiary_bank_acquirer", "status", "balance_before",
            "balance_after", "transaction_type_desc", "credit_acct_no", "debit_acct_no",
        ],
        "customer_profile_columns": [
            "account_open_date", "account_type", "customer_segment",
            "branch_code_profile", "city", "occupation", "yearly_income",
            "risk_rating", "account_age_days", "account_age_years", "new_account_indicator",
        ],
        "pending_feature_groups": ["device", "risk_indicators", "peer_group", "cash_transactions"],
        "pending_note": pending_note,
        "ready_numeric_features": [f for f in ready if f not in demo_cat],
        "demographic_numeric": demo_num,
        "demographic_flags": demo_flag,
        "demographic_categorical": demo_cat,
    }


def parse_feature_store(path: Path) -> list[dict]:
    xl = pd.ExcelFile(path)
    definitions: list[dict] = []
    skip_sheets = {"Device", "Rules-Risk Indicators", "cash transactions", "Peer Group Features"}

    for sheet in xl.sheet_names:
        if sheet in skip_sheets:
            continue
        df = pd.read_excel(path, sheet_name=sheet)
        cols = [str(c).lower() for c in df.columns]
        name_col = next(
            (df.columns[i] for i, c in enumerate(cols) if "feature" in c and "name" in c),
            df.columns[0],
        )
        desc_col = next(
            (df.columns[i] for i, c in enumerate(cols) if "desc" in c or "definition" in c),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        formula_col = next(
            (df.columns[i] for i, c in enumerate(cols)
             if "formula" in c or "logic" in c or "sql" in c or "pseudo" in c),
            None,
        )
        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            if not name or name.lower() in ("feature name", "nan", "z-score calculation"):
                continue
            if not re.match(r"^[a-z0-9_]+", name.lower().replace(" ", "")):
                continue
            definitions.append({
                "sheet": sheet,
                "feature_name": name,
                "description": str(row.get(desc_col, "")) if desc_col else "",
                "formula": str(row.get(formula_col, "")) if formula_col else "",
            })
    return definitions


def main() -> None:
    if not COLUMN_LIST.exists():
        print(f"Column list not found: {COLUMN_LIST}")
        sys.exit(1)
    if not FEATURE_STORE.exists():
        print(f"Feature Store not found: {FEATURE_STORE}")
        sys.exit(1)

    registry = parse_column_list(COLUMN_LIST)
    definitions = parse_feature_store(FEATURE_STORE)

    config_dir = ROOT / "config"
    config_dir.mkdir(exist_ok=True)

    registry_path = config_dir / "feature_registry.yaml"
    with open(registry_path, "w", encoding="utf-8") as f:
        yaml.dump(registry, f, default_flow_style=False, sort_keys=False)

    defs_path = config_dir / "feature_definitions.json"
    defs_path.write_text(json.dumps(definitions, indent=2), encoding="utf-8")

    print(f"Wrote {registry_path} ({len(registry['ready_numeric_features'])} ready features)")
    print(f"Wrote {defs_path} ({len(definitions)} feature definitions)")


if __name__ == "__main__":
    main()
