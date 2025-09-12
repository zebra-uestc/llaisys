import os
import subprocess
import sys

TEST_OPS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "ops"))
os.chdir(TEST_OPS_DIR)


def run_tests(args):
    failed = []
    for test in [
        "add.py",
        "argmax.py",
        "embedding.py",
        "linear.py",
        "rms_norm.py",
        "rope.py",
        "self_attention.py",
        "swiglu.py",
    ]:
        result = subprocess.run(
            f"python {test} {args}", text=True, encoding="utf-8", shell=True
        )
        if result.returncode != 0:
            failed.append(test)

    return failed


if __name__ == "__main__":
    failed = run_tests(" ".join(sys.argv[1:]))
    if len(failed) == 0:
        print("\033[92mAll tests passed!\033[0m")
    else:
        print("\033[91mThe following tests failed:\033[0m")
        for test in failed:
            print(f"\033[91m - {test}\033[0m")
    exit(len(failed))
