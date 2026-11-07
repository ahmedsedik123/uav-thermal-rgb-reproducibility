"""Check the archived benchmark inputs and the release evidence without loading weights."""
import argparse
import csv
import hashlib
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
p.add_argument("--records-only", action="store_true", help="Verify records before extracting the binary inputs.")
args = p.parse_args()
root = args.root.resolve()

def table(name):
    with (root / "records" / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

index = table("run_checkpoint_index.csv")
assert len(index) == 60 and len({(x["cohort"],x["model"],x["seed"]) for x in index}) == 60
for e in index:
    assert (root / e["run_record"]).is_file(), e["run_record"]
    hist = (root / e["run_record"]).with_name("history.csv")
    with hist.open(encoding="utf-8", newline="") as f:
        assert len(list(csv.DictReader(f))) == 100, hist
for e in table("record_file_digests.csv"):
    assert digest(root / e["path"]) == e["release_sha256"], e["path"]
if not args.records_only:
    for e in table("input_files.csv"):
        f = root / e["path"]
        assert f.stat().st_size == int(e["bytes"]), f
        assert digest(f) == e["sha256"], f
    print("Verified archived images, annotations, and initial weights.")
print("Verified 60 run records and recorded histories.")
