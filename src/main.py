"""CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.data.database import LocalDatabase
from src.data.preprocessor import clean_data
from src.data.synthetic import generate_master_csv
from src.pipeline import evaluate_all, train_all


def cmd_db_status(_: argparse.Namespace) -> None:
    db = LocalDatabase()
    import json
    print(json.dumps(db.summary(), indent=2))


def cmd_import(args: argparse.Namespace) -> None:
    db = LocalDatabase()
    path = Path(args.input)
    df = db.import_csv(path, overwrite=args.overwrite)
    print(f"Imported {df.height} rows, {df.width} columns → {db.path}")


def cmd_generate(args: argparse.Namespace) -> None:
    df = generate_master_csv(
        n_accounts=args.accounts,
        txns_per_account=args.txns_per_account,
        save=True,
    )
    print(f"Generated {df.height} rows, {df.width} columns → {settings.database_path}")


def cmd_train(_: argparse.Namespace) -> None:
    results = train_all()
    for name, metrics in results.items():
        label = {
            "isolation_forest": "Isolation Forest",
            "lof": "LOF",
            "ocsvm": "One-Class SVM",
            "ae": "Autoencoder",
        }.get(name, name)
        print(f"[{label}] anomaly_rate_test={metrics['anomaly_rate_test']:.4f}")


def cmd_evaluate(_: argparse.Namespace) -> None:
    path = evaluate_all()
    print(f"Evaluation report → {path}")


def cmd_parse_excel(_: argparse.Namespace) -> None:
    import subprocess
    script = ROOT / "scripts" / "parse_excel_specs.py"
    subprocess.run([sys.executable, str(script)], check=True)


def cmd_clean_data(args: argparse.Namespace) -> None:
    import polars as pl

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(f"{input_path.stem}_cleaned")
    df = pl.read_csv(input_path, try_parse_dates=True, infer_schema_length=10_000)
    cleaned = clean_data(df)
    cleaned.write_csv(output_path)
    print(f"Cleaned {df.height} rows -> {cleaned.height} rows → {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Banking Anomaly Detection Pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_db = sub.add_parser("db-status", help="Show local database summary")
    p_db.set_defaults(func=cmd_db_status)

    p_imp = sub.add_parser("import-csv", help="Import CSV into local database")
    p_imp.add_argument("--input", required=True, help="Source CSV path")
    p_imp.add_argument("--overwrite", action="store_true")
    p_imp.set_defaults(func=cmd_import)

    p_gen = sub.add_parser("generate-synthetic", help="Generate synthetic master CSV")
    p_gen.add_argument("--accounts", type=int, default=50)
    p_gen.add_argument("--txns-per-account", type=int, default=100)
    p_gen.set_defaults(func=cmd_generate)

    p_train = sub.add_parser("train", help="Train IF, LOF, OCSVM, Autoencoder")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Run unsupervised evaluation")
    p_eval.set_defaults(func=cmd_evaluate)

    p_parse = sub.add_parser("parse-excel", help="Regenerate feature_registry from Excel specs")
    p_parse.set_defaults(func=cmd_parse_excel)

    p_clean = sub.add_parser("clean-data", help="Clean and standardize a raw transaction CSV")
    p_clean.add_argument("--input", required=True, help="Source CSV path")
    p_clean.add_argument("--output", help="Optional output CSV path")
    p_clean.set_defaults(func=cmd_clean_data)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
