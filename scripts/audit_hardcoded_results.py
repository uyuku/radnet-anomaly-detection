#!/usr/bin/env python3
"""
scripts/audit_hardcoded_results.py

Comprehensive repository-wide audit for hardcoded results vs. verified constants.
Audits every results-generating script in scripts/ and src/:
1. Confirms each results-writing script reads raw or processed input data.
2. Identifies literal numbers in dictionaries, lists, and tuples.
3. Categorizes each hit as:
   - REAL_CONSTANT (physical constant, half-life, branching ratio, nominal geometry)
   - AUTHORIZED_CONFIG (grid parameter, threshold sweep range, random seed, plot styling)
   - HARDCODED_RESULT (unauthorized static result numbers presented as computed)

Outputs:
- docs/hardcode_audit.csv
"""

import ast
from pathlib import Path
import sys
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

# Physics constants keywords and known constants
KNOWN_PHYSICS_KEYWORDS = {
    "half_life", "lambda", "t_half", "branching", "gamma_energy", "kev", "mev",
    "fgr12", "icrp", "dose_coeff", "submersion", "flow_rate", "cfm", "attenuation",
    "detector_efficiency", "photopeak", "radius", "volume", "density", "channels"
}

# Config keywords
KNOWN_CONFIG_KEYWORDS = {
    "seed", "alpha", "dpi", "figsize", "bins", "grid", "thresholds", "taus",
    "color", "linestyle", "linewidth", "lw", "marker", "percentiles", "xlim", "ylim"
}


def audit_repository():
    print("=" * 70)
    print("REPOSITORY HARDCODE & DATA-READING AUDIT")
    print("=" * 70)

    scripts = sorted(list((REPO_ROOT / "scripts").glob("*.py")) + list((REPO_ROOT / "src").glob("*.py")))
    audit_records = []
    scripts_without_data_reading = []

    data_read_methods = {"read_csv", "read_parquet", "load", "read_text", "open", "read_feather", "urlopen", "requests"}
    simulation_generator_modules = {"src/detector_response.py"}

    for s_path in scripts:
        code_text = s_path.read_text(encoding="utf-8")
        rel_path = str(s_path.relative_to(REPO_ROOT))
        tree = ast.parse(code_text, filename=str(s_path))

        # Check if script writes results (contains to_csv, write_text, etc.)
        writes_results = any(
            w in code_text for w in ["to_csv", ".to_parquet", "write_text", "json.dump", "savefig"]
        )
        has_data_read = any(r in code_text for r in data_read_methods)

        if writes_results and not has_data_read and rel_path not in simulation_generator_modules:
            # Check if it imports a data reading module or function
            has_loader_import = any(
                term in code_text for term in ["_prepare_all_station_data", "load_dataset", "compute_features"]
            )
            if not has_loader_import:
                scripts_without_data_reading.append(rel_path)

        # Scan for AST literal collections
        for node in ast.walk(tree):
            if isinstance(node, (ast.Dict, ast.List)):
                lineno = getattr(node, "lineno", 0)
                # Extract float constants
                floats = [
                    el.value
                    for el in ast.walk(node)
                    if isinstance(el, ast.Constant) and isinstance(el.value, float)
                ]
                if not floats:
                    continue

                # Context lines around node
                lines = code_text.splitlines()
                ctx_line = lines[lineno - 1].strip() if 0 < lineno <= len(lines) else ""

                # Classification heuristic
                ctx_lower = ctx_line.lower()
                is_phys = any(k in ctx_lower for k in KNOWN_PHYSICS_KEYWORDS) or "fukushima" in ctx_lower or "physics" in ctx_lower
                is_cfg = any(k in ctx_lower for k in KNOWN_CONFIG_KEYWORDS) or "grid" in ctx_lower or "taus" in ctx_lower or "alpha" in ctx_lower

                # Detect hard-coded benchmark results (e.g. dictionaries with test_det_pct, fa_yr)
                is_hardcoded_result = False
                if isinstance(node, ast.Dict):
                    keys = [
                        k.value for k in node.keys
                        if isinstance(k, ast.Constant) and isinstance(k.value, str)
                    ]
                    result_keys = {"test_det_pct", "realized_test_fa", "det_ci_95", "fa_ci_95"}
                    if set(keys).intersection(result_keys):
                        is_hardcoded_result = True

                if is_hardcoded_result:
                    cat = "UNAUTHORIZED_HARDCODED_RESULT"
                elif is_phys:
                    cat = "REAL_PHYSICS_CONSTANT"
                elif is_cfg:
                    cat = "AUTHORIZED_CONFIG_GRID"
                else:
                    cat = "PARAMETER_OR_CONSTANT"

                audit_records.append({
                    "file": str(rel_path),
                    "line": lineno,
                    "category": cat,
                    "code_snippet": ctx_line[:100],
                    "sample_floats": str(floats[:3]),
                })

    df_audit = pd.DataFrame(audit_records)
    out_csv = DOCS_DIR / "hardcode_audit.csv"
    df_audit.to_csv(out_csv, index=False)
    print(f"Audited {len(scripts)} scripts. Logged {len(df_audit)} literal entries to {out_csv}")

    unauthorized = df_audit[df_audit["category"] == "UNAUTHORIZED_HARDCODED_RESULT"]
    if not unauthorized.empty:
        print(f"\nCRITICAL WARNING: Found {len(unauthorized)} unauthorized hard-coded results!")
        for _, r in unauthorized.iterrows():
            print(f"  - {r['file']}:{r['line']} -> {r['code_snippet']}")
        return False
    else:
        print("Verification PASSED: 0 unauthorized hardcoded results found.")

    if scripts_without_data_reading:
        print(f"\nWARNING: Results-writing scripts without data reading: {scripts_without_data_reading}")
        return False
    else:
        print("Verification PASSED: All results-writing scripts perform verified data reading.")

    return True


if __name__ == "__main__":
    success = audit_repository()
    sys.exit(0 if success else 1)
