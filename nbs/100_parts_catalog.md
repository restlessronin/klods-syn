---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Parts Catalog

Build a working subset of LDraw parts for synthetic rendering,
using Rebrickable's database for categories and frequency data.

## Data sources

Rebrickable provides daily CSV dumps of the complete LEGO database:
parts, categories, set inventories, relationships, and element-to-design-ID mappings.

Download URL pattern: `https://cdn.rebrickable.com/media/downloads/{table}.csv.gz`

```python
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path("../data/rebrickable")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://cdn.rebrickable.com/media/downloads"

TABLES = [
    "parts",
    "part_categories",
    "inventory_parts",
    "inventories",
    "elements",
    "part_relationships",
]

EXCLUDE_CATEGORIES = {
    "Duplo, Quatro and Primo",
    "String, Bands and Reels",
    "Large Buildable Figures",
}

COVERAGE_TARGET = 0.90


def fetch_table(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / f"{name}.csv.gz"
    if not path.exists():
        import urllib.request

        urllib.request.urlretrieve(f"{BASE_URL}/{name}.csv.gz", path)
    return pd.read_csv(path)
```

## Load tables

```python
parts = fetch_table("parts")
categories = fetch_table("part_categories")
inv_parts = fetch_table("inventory_parts")
inventories = fetch_table("inventories")
elements = fetch_table("elements")
relationships = fetch_table("part_relationships")

for name, df in [
    ("Parts", parts),
    ("Categories", categories),
    ("Inventory entries", inv_parts),
    ("Elements", elements),
    ("Relationships", relationships),
]:
    print(f"{name}: {len(df):,} — {df.columns.tolist()}")
```

## Part frequency across sets

How often each part appears across all official LEGO sets.
Parts that appear in more sets are more likely to be encountered
when sorting a real collection.

```python
set_inventories = inventories[inventories["version"] == 1]

part_freq = (
    inv_parts[~inv_parts["is_spare"]]
    .merge(set_inventories[["id", "set_num"]], left_on="inventory_id", right_on="id")
    .groupby("part_num")
    .agg(n_sets=("set_num", "nunique"), total_qty=("quantity", "sum"))
    .sort_values("n_sets", ascending=False)
)

print(f"Unique parts in sets: {len(part_freq):,}")
part_freq.head(20)
```

## Resolve LDraw geometry

LDraw uses LEGO Design IDs as filenames. Rebrickable's `part_num` often
matches directly, but not always. We resolve in three steps:

1. Direct match: `part_num` exists as an LDraw `.dat` file
2. Design ID from `elements.csv`: the `design_id` column maps to LDraw's numbering
3. Mold/alternate relationships from `part_relationships.csv`

```python
LDRAW_DIR = Path("../data/ldraw/parts")

ldraw_files = {p.stem for p in LDRAW_DIR.glob("*.dat")}
print(f"LDraw .dat files: {len(ldraw_files):,}")
```

```python
# Step 1: direct match
direct = part_freq.index.intersection(ldraw_files)

# Step 2: design_id from elements table
design_ids = (
    elements[["part_num", "design_id"]]
    .dropna(subset=["design_id"])
    .drop_duplicates(subset=["part_num"])
    .set_index("part_num")["design_id"]
    .astype(str)
)

# Step 3: mold and alternate relationships
mold_alt = relationships[relationships["rel_type"].isin(["M", "A"])]
rel_map = pd.concat(
    [
        mold_alt[["child_part_num", "parent_part_num"]].rename(
            columns={"child_part_num": "part_num", "parent_part_num": "ldraw_candidate"}
        ),
        mold_alt[["parent_part_num", "child_part_num"]].rename(
            columns={"parent_part_num": "part_num", "child_part_num": "ldraw_candidate"}
        ),
    ]
).drop_duplicates()


def resolve_ldraw(part_num: str) -> str | None:
    if part_num in ldraw_files:
        return part_num
    did = design_ids.get(part_num)
    if did and did in ldraw_files:
        return did
    candidates = rel_map.loc[rel_map["part_num"] == part_num, "ldraw_candidate"]
    for c in candidates:
        if c in ldraw_files:
            return c
        c_did = design_ids.get(c)
        if c_did and c_did in ldraw_files:
            return c_did
    return None


ldraw_resolved = pd.Series(
    {pn: resolve_ldraw(pn) for pn in part_freq.index}, name="ldraw_id"
)

matched = ldraw_resolved.dropna()
print(f"Direct match:        {len(direct):,}")
print(f"After full resolve:  {len(matched):,}")
print(f"Unmatched:           {len(part_freq) - len(matched):,}")
```

## Unmatched parts analysis

Inspect top unmatched parts to understand what's missing.
Many are printed variants or assemblies that share geometry
with a base part we already have.

```python
unmatched = part_freq.loc[part_freq.index.difference(matched.index)]
unmatched_top = (
    unmatched.sort_values("total_qty", ascending=False)
    .head(30)
    .merge(parts, left_index=True, right_on="part_num")
    .merge(categories, left_on="part_cat_id", right_on="id", suffixes=("", "_cat"))[
        ["part_num", "name", "name_cat", "n_sets", "total_qty"]
    ]
)
unmatched_top
```

## Coverage with base-part matching

Printed variants (`pr`), patterns (`pat`), and assemblies (`c01`)
share geometry with their base part. We count their quantity toward
the base part's LDraw geometry without adding to the render count.

```python
def strip_to_base(part_num: str) -> str:
    """Strip print, pattern, and assembly suffixes to get the base part number."""
    s = re.sub(r"pr\d+$", "", part_num)
    s = re.sub(r"pat\d+$", "", s)
    s = re.sub(r"c\d+$", "", s)
    return s


def resolve_ldraw_with_base(part_num: str) -> tuple[str | None, str]:
    """Returns (ldraw_id, match_type) where match_type is 'exact' or 'base'."""
    exact = resolve_ldraw(part_num)
    if exact:
        return exact, "exact"
    base = strip_to_base(part_num)
    if base != part_num:
        base_resolved = resolve_ldraw(base)
        if base_resolved:
            return base_resolved, "base"
    return None, "none"


results = pd.DataFrame(
    [resolve_ldraw_with_base(pn) for pn in part_freq.index],
    index=part_freq.index,
    columns=["ldraw_id", "match_type"],
)

counts = results["match_type"].value_counts()
print(f"Exact:     {counts.get('exact', 0):,}")
print(f"Base:      {counts.get('base', 0):,}")
print(f"Unmatched: {counts.get('none', 0):,}")
```

## Cumulative coverage

Each distinct LDraw geometry accumulates quantity from all parts
that resolve to it (exact matches + base variants).
Coverage is measured against all parts, so percentages reflect
real-world coverage of a bulk collection.

```python
all_matched = results[results["match_type"] != "none"].copy()
all_matched["total_qty"] = part_freq["total_qty"]

ldraw_qty = (
    all_matched.groupby("ldraw_id")["total_qty"].sum().sort_values(ascending=False)
)

total = part_freq["total_qty"].sum()
coverage = ldraw_qty.cumsum() / total

print(f"Distinct LDraw geometries: {ldraw_qty.shape[0]:,}")
for target in [0.80, 0.85, 0.90, 0.95, 0.99]:
    n = (coverage <= target).sum() + 1
    print(f"{target:.0%} coverage: {n:,} LDraw parts to render")

coverage.reset_index(drop=True).plot(
    title="Cumulative coverage (distinct LDraw geometries, vs all parts)",
    xlabel="Number of LDraw parts to render",
    ylabel="Fraction of all pieces",
    figsize=(10, 4),
)
```

## Category breakdown of selected parts

```python
cutoff = (coverage <= COVERAGE_TARGET).sum() + 1
selected_ldraw = ldraw_qty.index[:cutoff]

exact_matched = results[results["match_type"] == "exact"]
rep_parts = (
    part_freq.loc[exact_matched.index]
    .reset_index()
    .assign(ldraw_id=exact_matched.loc[exact_matched.index, "ldraw_id"].values)
    .sort_values("total_qty", ascending=False)
    .drop_duplicates(subset=["ldraw_id"])
    .set_index("ldraw_id")
)

selected_with_cats = (
    rep_parts.loc[rep_parts.index.intersection(selected_ldraw)]
    .reset_index()
    .merge(parts, on="part_num")
    .merge(categories, left_on="part_cat_id", right_on="id", suffixes=("", "_cat"))
)

selected_with_cats.groupby("name_cat").size().sort_values(ascending=False)
```

## Export working subset

Select parts by coverage threshold, excluding categories that
can't be meaningfully rendered (Duplo, string/rubber, large figures).

```python
output = selected_with_cats[~selected_with_cats["name_cat"].isin(EXCLUDE_CATEGORIES)][
    ["ldraw_id", "part_num", "name", "name_cat", "n_sets", "total_qty"]
].copy()
output.columns = [
    "ldraw_id",
    "part_num",
    "part_name",
    "category",
    "n_sets",
    "total_qty",
]
output = output.sort_values("total_qty", ascending=False)

excluded = selected_with_cats[selected_with_cats["name_cat"].isin(EXCLUDE_CATEGORIES)]
print(f"Excluded {len(excluded)} parts from: {set(excluded['name_cat'])}")

out_path = DATA_DIR / "working_subset.csv"
output.to_csv(out_path, index=False)
print(
    f"Exported {len(output):,} LDraw parts ({COVERAGE_TARGET:.0%} coverage) to {out_path}"
)
```
