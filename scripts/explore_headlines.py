"""Headline structure: company names, templates, companies per session, recurrence."""
import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
pd.set_option("display.width", 250)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.max_rows", 400)

h = pd.read_parquet(DATA / "ins_headlines.parquet")
ho = pd.read_parquet(DATA / "oos_headlines.parquet")

# Company = first two tokens? Check token-2 vocabulary (sector words) and token-1 shape.
tok = h["headline"].str.split()
print("token[1] (sector word?) top values:")
print(tok.str[1].value_counts().head(40))
print("n distinct token[1]:", tok.str[1].nunique())

h["company"] = tok.str[:2].str.join(" ")
h["rest"] = tok.str[2:].str.join(" ")
print("\ndistinct companies IS:", h["company"].nunique())
print("companies per session:")
print(h.groupby("session")["company"].nunique().value_counts().sort_index())


def templ(s: str) -> str:
    s = re.sub(r"\$[\d.,]+[MBK]?", "$X", s)
    s = re.sub(r"\d+(\.\d+)?%", "N%", s)
    s = re.sub(r"\b\d+(\.\d+)?\b", "N", s)
    return s


h["templ"] = h["rest"].map(templ)
print("\ndistinct templates (after number masking):", h["templ"].nunique())
print(h["templ"].value_counts().head(60))

# Do companies recur across sessions?
sess_per_co = h.groupby("company")["session"].nunique()
print("\nsessions per company:")
print(sess_per_co.describe())

# OOS overlap
ho_tok = ho["headline"].str.split()
ho["company"] = ho_tok.str[:2].str.join(" ")
ov = set(ho["company"]) & set(h["company"])
print("\nOOS companies:", ho["company"].nunique(), " overlap with IS:", len(ov))

h.to_pickle(Path(__file__).resolve().parents[1] / "scratch_h.pkl")
