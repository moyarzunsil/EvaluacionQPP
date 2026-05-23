import json
import math
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


def normal_cdf(z: float) -> float:
    """Approximate standard normal CDF Φ(z)."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def kendall_pvalue(tau: float, n: int) -> float:
    """Approximate two-sided p-value for Kendall's tau given n (large-sample)."""
    if n <= 0:
        return 1.0
    # Asymptotic variance under H0 (no ties)
    var0 = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    if var0 <= 0:
        return 1.0
    z = tau / math.sqrt(var0)
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    return max(min(p, 1.0), 0.0)


def parse_kendall_section(text: str):
    """Parse Kendall section of qpp_correlation_report.txt for one dataset.

    Returns:
        n_queries: int
        taus: dict[method] -> (tau_ndcg, tau_ap)
    """
    m = re.search(r"Correlations KENDALL:(.*?)(?:\nCorrelations SPEARMAN:|\Z)", text, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Could not find Correlations KENDALL section")
    block = m.group(1)

    # Number of queries: we use the one from IDF Average as representative
    m_n = re.search(r"IDF Average:\s*(\d+)\s+queries", block)
    if not m_n:
        raise RuntimeError("Could not find the number of queries (IDF Average)")
    n_queries = int(m_n.group(1))

    # Correlation values table
    m_tab = re.search(
        r"Correlation values .*?\n(.*?)(?:\n\n|Summary Statistics:)",
        block,
        flags=re.DOTALL,
    )
    if not m_tab:
        raise RuntimeError("Could not find Kendall values table")
    table = m_tab.group(1)

    lines = [ln.strip() for ln in table.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("Kendall table is empty")

    # First line is header: 'ndcg@10 ap'
    taus = {}
    for line in lines[1:]:
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
        ap = float(m_row.group("ap"))
        taus[method] = (ndcg, ap)

    return n_queries, taus


def main():
    results = {}

    for ds in DATASETS:
        path = BASE / ds / "qpp_correlation_report.txt"
        text = path.read_text(encoding="utf-8")
        n, taus = parse_kendall_section(text)
        res_ds = {}
        for method in METHODS:
            tau_ndcg, tau_ap = taus[method]
            p_ndcg = kendall_pvalue(tau_ndcg, n)
            p_ap = kendall_pvalue(tau_ap, n)
            res_ds[method] = {
                "n": n,
                "tau_ndcg": tau_ndcg,
                "tau_ap": tau_ap,
                "p_ndcg": p_ndcg,
                "p_ap": p_ap,
            }
        results[ds] = res_ds

    # Averages per method across datasets
    avg = {}
    for method in METHODS:
        p_ndcg_vals = [results[ds][method]["p_ndcg"] for ds in DATASETS]
        p_ap_vals = [results[ds][method]["p_ap"] for ds in DATASETS]
        avg[method] = {
            "p_ndcg_mean": sum(p_ndcg_vals) / len(p_ndcg_vals),
            "p_ap_mean": sum(p_ap_vals) / len(p_ap_vals),
        }

    print("# Summary of Kendall p-values (ndcg@10 and ap) by dataset and method\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))

    print("\n# Average of p-values per method (for significance table)\n")
    print(json.dumps(avg, indent=2, ensure_ascii=False))

    print("\n# Suggested Typst rows for the significance table (mean of p)\n")
    for method in METHODS:
        p_ndcg = avg[method]["p_ndcg_mean"]
        p_ap = avg[method]["p_ap_mean"]

        def fmt(p: float) -> str:
            if p < 0.001:
                return "<0.001"
            if p < 0.01:
                return f"{p:.3f}"
            if p < 0.1:
                return f"{p:.2f}"
            return f"{p:.1f}"

        print(
            f"    [*{method.replace(' ', ' ')}*],"
            f"[$ {fmt(p_ndcg)} $],"
            f"[$ {fmt(p_ap)} $],"
        )


if __name__ == "__main__":
    main()


