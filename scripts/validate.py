import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_step(name, args, input_text=None, must_contain=None):
    print(f"\n== {name} ==")
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        input=input_text,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        print(output)
        raise SystemExit(f"{name} failed with exit code {proc.returncode}")
    for expected in must_contain or []:
        if expected not in output:
            print(output)
            raise SystemExit(f"{name} did not contain expected text: {expected}")
    print("OK")
    return output


def python_files():
    folders = ["examples", "benchmarks", "tests", "scripts", "experiments"]
    files = [ROOT / "tree_memory_engine.py"]
    for folder in folders:
        files.extend(sorted((ROOT / folder).rglob("*.py")))
    return [str(path.relative_to(ROOT)) for path in files]


def main():
    files = python_files()
    run_step("syntax", ["-m", "py_compile", *files])
    run_step("unit tests", ["-m", "unittest", "discover", "-s", "tests"], must_contain=["OK"])
    run_step(
        "demo",
        ["examples/demo.py"],
        must_contain=[
            "Who produces premium car tires? -> Michelin",
            "Who produces premium car tires now? -> Bridgestone",
            "What are Michelin stars, not tires? -> restaurant awards",
        ],
    )
    run_step(
        "interactive cli smoke",
        ["examples/interactive_cli.py"],
        input_text="ask Who produces premium car tires?\nask What are Michelin stars?\nquit\n",
        must_contain=["Answer: Michelin", "Answer: restaurant awards"],
    )
    run_step(
        "benchmark",
        ["benchmarks/flat_vs_tree_5tasks.py"],
        must_contain=["Final Best Tree vs Flat verdict: PASS"],
    )
    run_step(
        "scaled benchmark",
        ["benchmarks/scaled_memory_benchmark.py"],
        must_contain=["Final Scaled Memory Benchmark verdict: PASS"],
    )
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
