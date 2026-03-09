"""CLI interface for rag-doctor (stdlib argparse)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from .doctor import Doctor
from .config import RagDoctorConfig


def _make_doctor(config: Optional[str]) -> Doctor:
    if config and Path(config).exists():
        cfg = RagDoctorConfig.from_yaml(config)
    else:
        cfg = RagDoctorConfig.default()
    return Doctor(cfg)


SEVER_COLOR = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[91m", "critical": "\033[31m"}
RESET = "\033[0m"


def _print_report(report, use_color=True):
    c = SEVER_COLOR.get(report.severity, "") if use_color else ""
    r = RESET if use_color else ""
    print("=" * 62)
    print(f"  RAG-DOCTOR {'✓ HEALTHY' if report.passed else '✗ ISSUES FOUND'}")
    print("=" * 62)
    print(f"  Root Cause : {report.root_cause} ({report.root_cause_id})")
    print(f"  Severity   : {c}{report.severity.upper()}{r}")
    print(f"  Finding    : {report.finding}")
    print("-" * 62)
    print("  Tool Results:")
    for tr in report.tool_results:
        tc = SEVER_COLOR.get(tr.severity, "") if use_color else ""
        st = "✓" if tr.passed else "✗"
        print(f"    {st} [{tc}{tr.severity:8}{r}] {tr.tool_name}: {tr.finding}")
    if report.fix_suggestion:
        print("-" * 62)
        print(f"  Fix: {report.fix_suggestion}")
    if report.config_patch:
        print(f"  Config Patch: {json.dumps(report.config_patch)}")
    print("=" * 62)


def cmd_diagnose(args):
    doctor = _make_doctor(args.config)
    report = doctor.diagnose(
        query=args.query, answer=args.answer,
        expected=getattr(args, "expected", None),
        corpus_texts=None,
    )
    if args.output == "json":
        print(report.to_json())
    else:
        _print_report(report)
    return 0 if report.passed else 1


def cmd_batch(args):
    doctor = _make_doctor(args.config)
    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {args.input}", file=sys.stderr)
        return 1
    cases = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    reports = doctor.batch_diagnose(cases)

    sev_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    threshold = sev_rank.get(getattr(args, "fail_on_severity", "high"), 2)

    print(f"\n{'#':>3}  {'Query':40}  {'Root Cause':25}  Severity")
    print("-" * 90)
    failures = 0
    for i, report in enumerate(reports, 1):
        c = SEVER_COLOR.get(report.severity, "")
        status = "✓" if report.passed else "✗"
        print(f"{i:>3}  {report.query[:40]:40}  {report.root_cause:25}  {c}{report.severity}{RESET}  {status}")
        if sev_rank.get(report.severity, 0) >= threshold:
            failures += 1

    print(f"\nTotal: {len(reports)} | Failures: {failures}")
    if args.output:
        Path(args.output).write_text(json.dumps([r.to_dict() for r in reports], indent=2))
        print(f"Report saved to {args.output}")
    return 1 if failures > 0 else 0


def main():
    parser = argparse.ArgumentParser(prog="rag-doctor", description="RAG pipeline failure diagnosis")
    sub = parser.add_subparsers(dest="command")

    p_diag = sub.add_parser("diagnose", help="Diagnose a single query")
    p_diag.add_argument("--query", "-q", required=True)
    p_diag.add_argument("--answer", "-a", required=True)
    p_diag.add_argument("--expected", "-e", default=None)
    p_diag.add_argument("--config", "-c", default=None)
    p_diag.add_argument("--output", "-o", default="text", choices=["text", "json"])

    p_batch = sub.add_parser("batch", help="Batch diagnose from JSONL")
    p_batch.add_argument("--input", "-i", required=True)
    p_batch.add_argument("--output", "-o", default=None)
    p_batch.add_argument("--config", "-c", default=None)
    p_batch.add_argument("--fail-on-severity", default="high")

    args = parser.parse_args()
    if args.command == "diagnose":
        sys.exit(cmd_diagnose(args))
    elif args.command == "batch":
        sys.exit(cmd_batch(args))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
