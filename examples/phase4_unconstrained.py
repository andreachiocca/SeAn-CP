"""Repeat Phase 4 without any specimen or test-machine assumptions.

Usage:
    python examples/phase4_unconstrained.py --n-cases 8000 --output results/phase4_search
"""
from seancp.unconstrained import main

if __name__ == '__main__':
    main()
