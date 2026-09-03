#!/usr/bin/env python3
"""Licences of the datasets the real-data arms read, as their own records state them.

WHY THIS IS A PROBE AND NOT A SENTENCE
The manuscript's data availability statement has to say under what licence each
dataset it read is distributed. Those licences live on the records: Zenodo's
metadata for the Monash collections, OpenML's per-dataset description for the two
tabular suites. A licence typed into a manuscript from memory is the same failure as
a number typed from memory, so this probe reads each record on a stated date and
writes what it found. The statement cites the records; this file is the dated
reading of them.

WHAT IS READ
  Monash.  The five collections `run_real_data.py`, `run_real_data_statsforecast.py`,
           `sample_robustness.py` and `frequency_and_model.py` load through sktime's
           `load_forecastingdata`, whose `tsf_dataset_names.py` maps each name to a
           Zenodo record. The record ids below are that mapping, copied with the
           sktime version that carried them named beside each. The API returns
           `metadata.license.id`, which is what is printed.
  OpenML.  Every dataset the tabular arm KEPT, read out of the committed
           `probe_output_real_data_tabular.txt` rather than re-listed here, so this
           probe cannot describe a dataset the arm did not use. The API's
           `data_set_description.licence` field is printed as the record gives it,
           including the value `Public`, which is OpenML's own label and not a licence
           name; the statement has to say that rather than translate it.

Nothing is derived. Counts at the foot are counts of the rows above them, and a
kept-list count that disagrees with the tabular output's own header aborts the run.

    python probes/dataset_licences.py

Needs the network and the standard library, nothing else.
"""

import datetime
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TABULAR = os.path.join(ROOT, "outputs", "probe_output_real_data_tabular.txt")
OUT = os.path.join(ROOT, "outputs", "probe_output_dataset_licences.txt")

# (dataset name as the loader takes it, Zenodo record id). The ids are sktime
# 1.1.0's `datasets/tsf_dataset_names.py`, the version environment [1] pins.
MONASH = [
    ("m1_monthly_dataset", 4656159),
    ("m1_quarterly_dataset", 4656154),
    ("m3_monthly_dataset", 4656298),
    ("m3_quarterly_dataset", 4656262),
    ("m4_daily_dataset", 4656548),
]

UA = {"User-Agent": "conformal-coverage-audit/dataset_licences (stdlib urllib)"}


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def kept_openml_ids():
    """The tabular arm's kept datasets, (kind, id, name), from its committed output.

    The output prints `<kind>: <N> datasets kept from <suite>` and then one indented
    row per dataset; the count on the header line is checked against the rows read,
    so a truncated block is caught here rather than reported as a smaller suite.
    """
    lines = open(TABULAR, encoding="utf-8").read().split("\n")
    kept, i = [], 0
    while i < len(lines):
        m = re.match(r"^(classification|regression): (\d+) datasets kept from ", lines[i])
        if not m:
            i += 1
            continue
        kind, n = m.group(1), int(m.group(2))
        rows = []
        i += 1
        while i < len(lines):
            r = re.match(r"^\s+(\d+)\s+(\S+)\s+(\d+) x (\d+)\s*$", lines[i])
            if not r:
                break
            rows.append((kind, int(r.group(1)), r.group(2)))
            i += 1
        assert len(rows) == n, f"{kind}: header says {n} kept, {len(rows)} rows read"
        kept += rows
    assert {k for k, _, _ in kept} == {"classification", "regression"}, (
        "the tabular output did not yield both kept blocks")
    return kept


def main():
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    L = []
    say = L.append
    say("=" * 96)
    say("DATASET LICENCES -- what each dataset's own record states, read on the date below")
    say("=" * 96)
    say(f"queried {today} UTC; Zenodo REST API for the Monash records, OpenML JSON API v1 for the suites")
    say("")
    say("MONASH TIME SERIES FORECASTING ARCHIVE (loaded through sktime.datasets.load_forecastingdata)")
    say(f"{'dataset':<24}{'zenodo record':>14}  {'title':<36}{'license id':<14}doi")
    monash_ids = []
    for name, rec in MONASH:
        d = get_json(f"https://zenodo.org/api/records/{rec}")
        md = d["metadata"]
        lic = md.get("license", {})
        lic_id = lic.get("id") if isinstance(lic, dict) else str(lic)
        doi = d.get("doi") or md.get("doi") or ""
        assert str(d.get("id", rec)) == str(rec), f"record {rec} answered as {d.get('id')}"
        say(f"{name:<24}{rec:>14}  {md.get('title', ''):<36}{str(lic_id):<14}{doi}")
        monash_ids.append(lic_id)
    say("")

    kept = kept_openml_ids()
    say("OPENML (the datasets the tabular arm kept; ids and names read from its committed output)")
    say(f"{'kind':<16}{'id':>8}  {'name':<32}{'version':>8}  licence")
    tally = {}
    for kind, did, name in kept:
        d = get_json(f"https://www.openml.org/api/v1/json/data/{did}")["data_set_description"]
        assert int(d["id"]) == did, f"OpenML {did} answered as {d['id']}"
        assert d["name"] == name, f"OpenML {did}: output says {name!r}, record says {d['name']!r}"
        lic = d.get("licence")
        lic = lic.strip() if isinstance(lic, str) and lic.strip() else "(blank)"
        say(f"{kind:<16}{did:>8}  {name:<32}{str(d.get('version', '')):>8}  {lic}")
        tally[lic] = tally.get(lic, 0) + 1
    say("")

    say("SUMMARY (counts of the rows above)")
    say(f"monash records                 {len(MONASH)}")
    say(f"monash records cc-by-4.0       {sum(1 for x in monash_ids if x == 'cc-by-4.0')}")
    say(f"openml datasets                {len(kept)}")
    say(f"openml classification          {sum(1 for k, _, _ in kept if k == 'classification')}")
    say(f"openml regression              {sum(1 for k, _, _ in kept if k == 'regression')}")
    for lic in sorted(tally):
        say(f"openml licence {lic:<48}{tally[lic]}")
    say("")
    say("OpenML's `Public` is the platform's own label for a dataset its uploader marked")
    say("freely usable; it names no licence text, and the statement reports it as the label")
    say("it is rather than as a licence. Every value above is the record's, not a reading of it.")

    body = "\n".join(L) + "\n"
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(body)
    print(body)
    print(f"written -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
