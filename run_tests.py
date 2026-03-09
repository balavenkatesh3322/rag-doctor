"""
Standalone test runner — no pytest required.
Runs all 62 tests and reports results.

Usage:
    python run_tests.py
    python run_tests.py --verbose
    python run_tests.py --unit-only
    python run_tests.py --e2e-only
"""

import sys
import os
import argparse
import traceback
import time

sys.path.insert(0, os.path.dirname(__file__))

# ─── Colour helpers ──────────────────────────────────────────────────────────
def c(t, code): return f"\033[{code}m{t}\033[0m"
PASS = c("PASS", "32"); FAIL = c("FAIL", "31")

results = {"passed": 0, "failed": 0, "errors": []}


def run_test(name: str, fn, verbose: bool = False):
    try:
        fn()
        if verbose:
            print(f"  {PASS}  {name}")
        else:
            print(".", end="", flush=True)
        results["passed"] += 1
    except AssertionError as e:
        print(f"\n  {FAIL}  {name}")
        tb = traceback.format_exc()
        # Find the actual assertion line
        for line in tb.splitlines():
            if "assert " in line or "AssertionError" in line:
                print(f"         {line.strip()}")
        results["failed"] += 1
        results["errors"].append((name, tb))
    except Exception as e:
        print(f"\n  {FAIL}  {name}")
        print(f"         {type(e).__name__}: {str(e)[:120]}")
        results["failed"] += 1
        results["errors"].append((name, traceback.format_exc()))


def collect_and_run(cls, verbose: bool = False):
    obj = cls()
    test_names = sorted(n for n in dir(cls) if n.startswith("test_"))
    if not test_names:
        return
    if verbose:
        print(f"\n  ── {cls.__name__} ──")
    else:
        print(f"\n  {cls.__name__:45}", end="")
    for name in test_names:
        fn = getattr(obj, name)
        if callable(fn):
            if hasattr(obj, "setup_method"):
                try: obj.setup_method()
                except Exception: pass
            run_test(f"{cls.__name__}.{name}", fn, verbose)
    if not verbose:
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--unit-only", action="store_true")
    parser.add_argument("--e2e-only", action="store_true")
    args = parser.parse_args()

    # Import test classes
    from tests.test_tools import (
        TestChunkAnalyzer, TestRetrievalAuditor, TestPositionTester,
        TestHallucinationTracer, TestChunkOptimizer, TestQueryRewriter,
    )
    from tests.test_connectors import TestMockConnector
    from tests.test_doctor import (
        TestScenario1ChunkFragmentation, TestScenario2PositionBias,
        TestScenario3Hallucination, TestScenario4RetrievalMiss,
        TestHealthyPipeline, TestBatchDiagnosis,
    )
    from tests.test_end_to_end import TestEndToEndScenarios, TestBatchEndToEnd

    print()
    print(c("═" * 58, "1"))
    print(c("  rag-doctor TEST SUITE", "1"))
    print(c("═" * 58, "1"))

    t0 = time.time()

    if not args.e2e_only:
        print("\n  [Unit Tests — Tools]")
        for cls in [TestChunkAnalyzer, TestRetrievalAuditor, TestPositionTester,
                    TestHallucinationTracer, TestChunkOptimizer, TestQueryRewriter]:
            collect_and_run(cls, args.verbose)

        print("\n  [Unit Tests — Connectors]")
        collect_and_run(TestMockConnector, args.verbose)

        print("\n  [Integration Tests — Scenarios]")
        for cls in [TestScenario1ChunkFragmentation, TestScenario2PositionBias,
                    TestScenario3Hallucination, TestScenario4RetrievalMiss,
                    TestHealthyPipeline, TestBatchDiagnosis]:
            collect_and_run(cls, args.verbose)

    if not args.unit_only:
        print("\n  [End-to-End Tests]")
        for cls in [TestEndToEndScenarios, TestBatchEndToEnd]:
            collect_and_run(cls, args.verbose)

    elapsed = time.time() - t0
    total = results["passed"] + results["failed"]

    print()
    print(c("═" * 58, "1"))
    color = "32" if results["failed"] == 0 else "31"
    print(c(f"  Results: {results['passed']}/{total} passed in {elapsed:.2f}s", color))

    if results["errors"]:
        print(f"\n  Failed tests:")
        for name, tb in results["errors"]:
            print(f"    ✗ {name}")
            for line in tb.splitlines():
                if "AssertionError" in line or "assert " in line:
                    print(f"      {line.strip()}")
                    break
    print(c("═" * 58, "1"))
    print()

    sys.exit(0 if results["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
