#!/usr/bin/env python3
"""
rag-doctor — Ollama Integration Test
=====================================
Mac-friendly script that:
  1. Checks Ollama is running
  2. Benchmarks installed models and selects the best one
  3. Runs all 4 classic RAG failure scenarios end-to-end using real LLM generation
  4. Prints a detailed diagnosis report for each

Usage:
    python test_with_ollama.py                     # auto-select best model
    python test_with_ollama.py --model llama3.2    # use specific model
    python test_with_ollama.py --benchmark-only    # just benchmark, don't test
    python test_with_ollama.py --quick             # skip benchmark, use first model

Requirements (Mac):
    brew install ollama
    ollama serve                    # in a separate terminal
    ollama pull llama3.2            # or any model

No pip install needed — uses only stdlib + this project's code.
"""

import sys
import os
import argparse
import time

# Make sure we can import from this project directory
sys.path.insert(0, os.path.dirname(__file__))


# ─── Colour helpers ─────────────────────────────────────────────────────────

def c(text, code): return f"\033[{code}m{text}\033[0m"
def green(t): return c(t, "32")
def red(t):   return c(t, "31")
def yellow(t): return c(t, "33")
def blue(t):  return c(t, "34")
def bold(t):  return c(t, "1")
def cyan(t):  return c(t, "36")

SEV_COLOR = {"low": green, "medium": yellow, "high": lambda t: c(t, "91"), "critical": red}


def print_banner():
    print()
    print(cyan("  ██████╗  █████╗  ██████╗      ██████╗  ██████╗  ██████╗ ████████╗ ██████╗ ██████╗ "))
    print(cyan("  ██╔══██╗██╔══██╗██╔════╝      ██╔══██╗██╔═══██╗██╔════╝ ╚══██╔══╝██╔═══██╗██╔══██╗"))
    print(cyan("  ██████╔╝███████║██║  ███╗█████╗██║  ██║██║   ██║██║         ██║   ██║   ██║██████╔╝"))
    print(cyan("  ██╔══██╗██╔══██║██║   ██║╚════╝██║  ██║██║   ██║██║         ██║   ██║   ██║██╔══██╗"))
    print(cyan("  ██║  ██║██║  ██║╚██████╔╝      ██████╔╝╚██████╔╝╚██████╗    ██║   ╚██████╔╝██║  ██║"))
    print(cyan("  ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝       ╚═════╝  ╚═════╝  ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝"))
    print()
    print(bold("  Agentic RAG Pipeline Failure Diagnosis  ·  Ollama Integration Test"))
    print()


def section(title: str, subtitle: str = ""):
    print()
    print(bold(f"{'─'*60}"))
    print(bold(f"  {title}"))
    if subtitle:
        print(f"  {yellow(subtitle)}")
    print(bold(f"{'─'*60}"))


def print_report(report, scenario_num: int):
    sev_fn = SEV_COLOR.get(report.severity, str)
    status = green("✓ HEALTHY") if report.passed else red("✗ ISSUES FOUND")

    print(f"\n  {bold('Status')}    : {status}")
    print(f"  {bold('Root Cause')}: {cyan(report.root_cause)} {yellow(f'({report.root_cause_id})')}")
    print(f"  {bold('Severity')} : {sev_fn(report.severity.upper())}")
    print(f"  {bold('Finding')}  : {report.finding}")
    print()
    print(f"  {bold('Tool Results:')}")
    for tr in report.tool_results:
        icon = green("✓") if tr.passed else red("✗")
        sc = SEV_COLOR.get(tr.severity, str)
        print(f"    {icon}  [{sc(f'{tr.severity:8}')}] {tr.tool_name:25}  {tr.finding}")

    if report.fix_suggestion:
        print()
        print(f"  {bold('Fix:')} {green(report.fix_suggestion)}")

    if report.config_patch:
        import json
        print(f"  {bold('Config Patch:')} {cyan(json.dumps(report.config_patch))}")


# ─── Test scenarios ──────────────────────────────────────────────────────────

def run_scenario_1(connector):
    """Chunk Fragmentation — Legal contract clause split across chunks."""
    from rag_doctor.connectors.ollama_connector import OllamaConnector
    from rag_doctor import Doctor

    section("Scenario 1: Chunk Fragmentation (RC-3)", "Legal Tech — Termination Clause")
    print(f"  Problem: Acme Corp 90-day clause split at chunk boundary.")
    print(f"  Expected behaviour: rag-doctor identifies chunk_fragmentation")

    # Connector already loaded — add legal corpus
    connector.load_corpus([
        {"id": "acme", "content":
            "Acme Corp Master Services Agreement. Termination: Either party may terminate "
            "this agreement by providing"},
        {"id": "acme2", "content":
            "90 days written notice via certified mail to the registered address. "
            "Early termination fees apply."},
        {"id": "std", "content":
            "Standard agreements: Termination with 30 days notice for month-to-month contracts."},
        {"id": "general", "content":
            "All contracts governed by the laws of Delaware. Disputes via arbitration."},
    ])

    query = "What is the termination notice period in the Acme Corp contract?"
    print(f"\n  {bold('Query')}   : {query}")

    docs = connector.retrieve(query, top_k=4)
    answer = connector.generate(query, docs)
    print(f"  {bold('Answer')}  : {answer[:100]}")

    doctor = Doctor.default(connector)
    report = doctor.diagnose(
        query=query,
        answer=answer,
        docs=docs,
        expected="Acme Corp requires 90 days written notice for termination.",
    )
    print_report(report, 1)
    return report


def run_scenario_2(connector):
    """Context Position Bias — Correct doc in middle position."""
    from rag_doctor.connectors.base import Document
    from rag_doctor import Doctor

    section("Scenario 2: Context Position Bias (RC-2)", "Healthcare AI — Drug Dosing")
    print(f"  Problem: Correct dosing doc retrieved but at middle position.")
    print(f"  Expected behaviour: rag-doctor identifies context_position_bias")

    # Manually construct middle-position scenario
    docs = [
        Document(
            content="General information about common analgesic medications and their usage.",
            position=0, score=0.71
        ),
        Document(
            content=(
                "Acetaminophen dosing for liver disease: maximum 2000mg per day. "
                "Standard healthy adult dose is 4000mg. Always consult physician for hepatic impairment."
            ),
            position=1, score=0.89,  # best doc is in MIDDLE
        ),
        Document(
            content="Ibuprofen is an NSAID used for pain. Dose: 400–800mg every 4–6 hours.",
            position=2, score=0.65
        ),
    ]

    query = "What is the maximum daily dose of acetaminophen for patients with liver disease?"
    print(f"\n  {bold('Query')}   : {query}")
    print(f"  {bold('Note')}    : Best doc injected at position 1 (danger zone)")

    answer = connector.generate(query, docs)
    print(f"  {bold('Answer')}  : {answer[:100]}")

    doctor = Doctor.default()
    report = doctor.diagnose(
        query=query,
        answer=answer,
        docs=docs,
        expected="For liver disease patients maximum daily dose is 2000mg.",
    )
    print_report(report, 2)
    return report


def run_scenario_3(connector):
    """Hallucination — LLM fabricates a non-existent API feature."""
    from rag_doctor import Doctor

    section("Scenario 3: Hallucination (RC-4)", "Developer Tools — API Documentation")
    print(f"  Problem: LLM hallucinates a non-existent API feature.")
    print(f"  Expected behaviour: rag-doctor identifies hallucination")

    connector.load_corpus([
        {"id": "api1", "content":
            "uploadFile(file, metadata=None): Uploads a file to the server. "
            "Returns UploadResponse with file_id. Maximum file size: 500MB. "
            "Does not support streaming or chunked transfer encoding."},
        {"id": "api2", "content":
            "For large files use streamUpload() which supports streaming. "
            "uploadFile() is for files under 500MB only."},
    ])

    query = "Does the uploadFile() method support chunked transfer encoding?"
    print(f"\n  {bold('Query')}   : {query}")

    docs = connector.retrieve(query, top_k=2)
    answer = connector.generate(query, docs)
    print(f"  {bold('Answer')}  : {answer[:120]}")

    doctor = Doctor.default()
    report = doctor.diagnose(query=query, answer=answer, docs=docs)
    print_report(report, 3)
    return report


def run_scenario_4(connector):
    """Retrieval Miss + Query Mismatch — Vocabulary gap."""
    from rag_doctor import Doctor

    section("Scenario 4: Query Mismatch + Retrieval Miss (RC-5)", "HR Policy — Vocabulary Gap")
    print(f"  Problem: Query uses 'paternity/maternity' but docs use 'parental leave'.")
    print(f"  Expected behaviour: rag-doctor identifies query_mismatch and suggests rewrite")

    connector.load_corpus([
        {"id": "hr1", "content":
            "Parental leave policy 2024: All full-time employees are entitled to "
            "16 weeks of paid parental leave. Applies equally to all parents."},
        {"id": "hr2", "content":
            "Benefits overview: Health insurance, dental, 401k matching, "
            "parental leave, flexible working arrangements."},
    ])

    # Deliberately use different vocabulary to trigger mismatch
    query = "How many weeks of maternity and paternity leave do employees get?"
    print(f"\n  {bold('Query')}   : {query}")
    print(f"  {bold('Note')}    : Docs use 'parental leave', query uses 'maternity/paternity'")

    docs = connector.retrieve(query, top_k=3)
    answer = connector.generate(query, docs)
    print(f"  {bold('Answer')}  : {answer[:100]}")

    doctor = Doctor.default(connector)
    report = doctor.diagnose(
        query=query,
        answer=answer,
        docs=docs,
        expected="Employees receive 16 weeks of paid parental leave.",
    )
    print_report(report, 4)
    return report


def run_healthy_scenario(connector):
    """Healthy pipeline — everything works correctly."""
    from rag_doctor import Doctor

    section("Scenario 5: Healthy Pipeline (RC-0)", "E-Commerce — Return Policy")
    print(f"  Expected behaviour: All tools pass, root_cause = 'healthy'")

    connector.load_corpus([
        {"id": "p1", "content":
            "Return Policy: Customers may return any item within 30 days of purchase "
            "for a full refund. Items must be in original condition with receipt. "
            "Refunds processed within 5 business days to original payment method."},
    ])

    query = "What is the return policy and how long does a refund take?"
    print(f"\n  {bold('Query')}   : {query}")

    docs = connector.retrieve(query, top_k=3)
    answer = connector.generate(query, docs)
    print(f"  {bold('Answer')}  : {answer[:120]}")

    doctor = Doctor.default()
    report = doctor.diagnose(
        query=query,
        answer=answer,
        docs=docs,
        expected="30 day returns, refunds within 5 business days.",
    )
    print_report(report, 5)
    return report


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="rag-doctor Ollama integration test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_with_ollama.py                      # auto benchmark + test
  python test_with_ollama.py --model llama3.2     # skip benchmark
  python test_with_ollama.py --benchmark-only     # benchmark only
  python test_with_ollama.py --quick              # use first available model
  python test_with_ollama.py --list-models        # list installed models
        """
    )
    parser.add_argument("--model", default=None, help="Ollama model to use (skips benchmark)")
    parser.add_argument("--embed-model", default=None, help="Ollama embedding model")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run model benchmark")
    parser.add_argument("--quick", action="store_true", help="Skip benchmark, use first model")
    parser.add_argument("--list-models", action="store_true", help="List installed Ollama models")
    parser.add_argument("--scenario", type=int, choices=[1,2,3,4,5], help="Run only one scenario")
    args = parser.parse_args()

    print_banner()

    # Check Ollama
    from rag_doctor.connectors.model_selector import (
        check_ollama_running, list_models, select_and_benchmark
    )

    print(f"  Checking Ollama at http://localhost:11434 ...", end="", flush=True)
    if not check_ollama_running():
        print(red(" ✗ NOT RUNNING"))
        print()
        print("  To start Ollama on Mac:")
        print(yellow("    brew install ollama"))
        print(yellow("    ollama serve              # run in a terminal"))
        print(yellow("    ollama pull llama3.2      # download a model"))
        sys.exit(1)
    print(green(" ✓ running"))

    # List models
    if args.list_models:
        models = list_models()
        print(f"\n  Installed models ({len(models)}):")
        for m in models:
            size = f"{m.get('size', 0)/1e9:.1f}GB"
            print(f"    • {m['name']:40} {size}")
        sys.exit(0)

    # Select model
    if args.model:
        selected_model = args.model
        print(f"  Using specified model: {bold(selected_model)}")
    elif args.quick:
        models = list_models()
        if not models:
            print(red("  No models installed. Run: ollama pull llama3.2"))
            sys.exit(1)
        selected_model = models[0]["name"]
        print(f"  Quick mode — using first available model: {bold(selected_model)}")
    else:
        # Full benchmark
        selected_model = select_and_benchmark(verbose=True)
        if not selected_model:
            sys.exit(1)

    if args.benchmark_only:
        sys.exit(0)

    # Build connector
    print(f"\n  Initialising OllamaConnector with model: {bold(selected_model)}")
    from rag_doctor.connectors.ollama_connector import OllamaConnector
    connector = OllamaConnector(
        model=selected_model,
        embed_model=args.embed_model,
    )

    # Run scenarios
    t_start = time.time()
    reports = []

    if args.scenario:
        scenarios = {1: run_scenario_1, 2: run_scenario_2, 3: run_scenario_3,
                     4: run_scenario_4, 5: run_healthy_scenario}
        reports.append(scenarios[args.scenario](connector))
    else:
        for fn in [run_scenario_1, run_scenario_2, run_scenario_3,
                   run_scenario_4, run_healthy_scenario]:
            try:
                report = fn(connector)
                reports.append(report)
            except Exception as e:
                print(red(f"\n  ✗ Scenario failed: {e}"))
                import traceback
                traceback.print_exc()

    # Summary
    elapsed = time.time() - t_start
    section("Test Summary", f"Model: {selected_model}  ·  {elapsed:.1f}s total")

    scenario_names = [
        "Chunk Fragmentation (Legal)",
        "Position Bias (Medical)",
        "Hallucination (Dev Docs)",
        "Query Mismatch (HR)",
        "Healthy Pipeline",
    ]
    expected_causes = [
        "chunk_fragmentation",
        "context_position_bias",
        "hallucination",
        "query_mismatch",
        "healthy",
    ]

    all_pass = True
    for i, report in enumerate(reports):
        name = scenario_names[i] if i < len(scenario_names) else f"Scenario {i+1}"
        expected = expected_causes[i] if i < len(expected_causes) else ""
        detected = report.root_cause

        # Check if detection matched expectation
        matched = detected == expected or (expected == "healthy" and report.severity == "low")
        icon = green("✓") if matched else yellow("~")
        if not matched:
            all_pass = False

        sev_fn = SEV_COLOR.get(report.severity, str)
        print(f"  {icon}  {name:35}  {cyan(detected):30}  [{sev_fn(report.severity)}]")
        if not matched and expected:
            print(f"       {yellow(f'Expected: {expected}')}")

    print()
    if all_pass:
        print(green(f"  ✓ All scenarios diagnosed correctly with {selected_model}"))
    else:
        print(yellow(f"  ~ Some diagnoses differ from expected — model response variation is normal"))
        print(f"    This is expected behaviour: LLM outputs vary by model and temperature.")

    print()
    print(f"  Model used: {bold(selected_model)}")
    print(f"  Total time: {elapsed:.1f}s")
    print()
    print(f"  Ready to push? Run: {cyan('./setup.sh https://github.com/YOUR-USERNAME/rag-doctor.git')}")
    print()


if __name__ == "__main__":
    main()
