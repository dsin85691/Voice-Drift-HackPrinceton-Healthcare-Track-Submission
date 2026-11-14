"""`python tools/budget.py --drive <Drive dir>`: print GPU hours used and remaining (spec 4)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.budget import main  # noqa: E402

if __name__ == "__main__":
    main()
