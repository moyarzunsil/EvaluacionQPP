import json
import re
from pathlib import Path


BASE = Path(__file__).resolve().parent

DATASETS = [
    "antique_test",
    "cranfield",
    "trec_covid",
    "msmarco_dl20_judged",
    "car_v15_trec_y1_manual",
]

METHODS = [
    "IDF Average",
    "IDF Maximum",
    "SCQ Average",
    "SCQ Maximum",
    "WIG",
    "NQC",
    "Clarity",
    "UEF-WIG",
    "UEF-NQC",
    "UEF-Clarity",
]

COR_TYPES = ["KENDALL", "SPEARMAN", "PEARSON"]


def parse_dataset(dataset: str):
    """Parse a single qpp_correlation_report.txt and extract ndcg@10 values."""
    path = BASE / dataset / "qpp_correlation_report.txt"
    text = path.read_text(encoding="utf-8")

    result = {m: {} for m in METHODS}

    for cor in COR_TYPES:
        m = re.search(
            rf"Correlations {cor}:(.*?)(?:\nCorrelations |\Z)",
            text,
            flags=re.DOTALL,
        )
        if not m:
            raise RuntimeError(f"Could not find section for {cor} in {dataset}")
        block = m.group(1)

        m_table = re.search(
            r"Correlation values[^\n]*\n(.*?)(?:\n\n|Summary Statistics)",
            block,
            flags=re.DOTALL,
        )
        if not m_table:
            raise RuntimeError(f"Could not find values table in {dataset} / {cor}")

        table = m_table.group(1)
        lines = [ln.strip() for ln in table.splitlines() if ln.strip()]
        if not lines:
            continue

        # First line: header "ndcg@10 ap"
        for line in lines[1:]:
            # Example line:
            # "IDF Average  0.096422  0.096616"
            m_row = re.match(
                r"(?P<method>[A-Za-zÁÉÍÓÚÜÑáéíóúüñ\- ]+?)\s+"
                r"(?P<ndcg>[+-]?\d+\.\d+)\s+"
                r"(?P<ap>[+-]?\d+\.\d+)",
                line,
            )
            if not m_row:
                continue
            method = m_row.group("method").strip()
            ndcg = float(m_row.group("ndcg"))
            if method in result:
                result[method][cor] = ndcg

    return result


def main():
    data = {ds: parse_dataset(ds) for ds in DATASETS}

    # Print raw JSON for debugging (ASCII only to avoid encoding issues)
    print("# Raw correlations JSON (ndcg@10):", flush=True)
    print(json.dumps(data, indent=2, ensure_ascii=True), flush=True)

    # Also generates Typst rows ready to paste into the table
    print("\n# Suggested Typst rows (ndcg@10, P-rho / S-rho / K-tau):\n", flush=True)

    # Logical order for correlation columns in the table
    cor_order = ["PEARSON", "SPEARMAN", "KENDALL"]

    for method in METHODS:
        parts = [f"  [{method}],"]
        for ds in DATASETS:
            vals = data[ds].get(method, {})
            for cor in cor_order:
                v = vals.get(cor)
                if v is None:
                    parts.append("  [$-$],")
                else:
                    parts.append(f"  [${v:.3f}$],")
        print(" ".join(parts))


if __name__ == "__main__":
    main()


