# test_ptb_load.py
from datasets import load_dataset

def try_load(desc, **kwargs):
    print(f"\n=== {desc} ===")
    try:
        ds = load_dataset(**kwargs)
        print("OK")
        print("num_rows:", len(ds))
        print("columns:", ds.column_names)
        print("first:", ds[0])
        print("second:", ds[1])
    except Exception as e:
        print("FAILED:", type(e).__name__, str(e))

if __name__ == "__main__":
    try_load(
        "script-style",
        path='wikitext',
        name='wikitext-2-raw-v1',
        split='test'
    )
