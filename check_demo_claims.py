"""
Run the demo claims through V2 evidence mode and compare with the expected verdicts.
Needs internet. Run before a demo to make sure everything still works:

    python check_demo_claims.py
"""

import sys
import time

from core.analyze import analyze
from core.schema import NOT_ENOUGH, REFUTED, SUPPORTED

DEMO_CLAIMS = [
    ("Chandrayaan-3 landed near the Moon's south pole in 2023", SUPPORTED),
    ("India became the most populous country in the world in 2023", SUPPORTED),
    ("Albert Einstein won the Nobel Prize in Physics for explaining the photoelectric effect", SUPPORTED),
    ("Mount Everest is the highest mountain on Earth above sea level", SUPPORTED),
    ("The Taj Mahal was built by Mughal emperor Shah Jahan", SUPPORTED),
    ("The new 2000 rupee notes contain a GPS nano chip", REFUTED),
    ("UNESCO declared Jana Gana Mana the best national anthem in the world", REFUTED),
    ("5G mobile networks spread the coronavirus", REFUTED),
    ("The Great Wall of China is visible from the Moon with the naked eye", REFUTED),
    ("A new species of bird was discovered in Pune last week", NOT_ENOUGH),
]


def main():
    verbose = "-v" in sys.argv
    passed = 0
    for claim, expected in DEMO_CLAIMS:
        start = time.time()
        r = analyze(claim, mode="evidence")
        ok = r["verdict"] == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {r['verdict'] or r['error']:<20} (expected {expected:<20}) "
              f"{time.time() - start:4.1f}s  {claim}")
        for w in r["warnings"]:
            print(f"        ! {w}")
        if verbose or not ok:
            for e in r["evidence"][:4]:
                print(f"        [{e['stance']:<8} {e['score']:.2f}] {e['source']}: {e['sentence'][:140]}")
    print(f"\n{passed}/{len(DEMO_CLAIMS)} demo claims match the expected verdict")


if __name__ == "__main__":
    main()
