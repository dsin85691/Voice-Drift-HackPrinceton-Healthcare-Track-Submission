"""`make artifacts ARTIFACT_URL=<direct zip url>`: download + unpack artifacts/vN from the private store."""
import io
import sys
import zipfile
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def main(url: str) -> None:
    r = httpx.get(url, follow_redirects=True, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for n in z.namelist():
            if n.startswith("/") or ".." in Path(n).parts:   # no path traversal
                raise SystemExit(f"refusing unsafe path in zip: {n}")
        z.extractall(ROOT / "artifacts")
    print("artifacts ready:", sorted(p.name for p in (ROOT / "artifacts").glob("v*")))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: fetch_artifacts.py <zip url>")
    main(sys.argv[1])
