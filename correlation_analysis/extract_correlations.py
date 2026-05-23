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
    "IDF Promedio",
    "IDF Máximo",
    "SCQ Promedio",
    "SCQ Máximo",
    "WIG",
    "NQC",
    "Clarity",
    "UEF-WIG",
    "UEF-NQC",
    "UEF-Clarity",
]

COR_TYPES = ["KENDALL", "SPEARMAN", "PEARSON"]


def parse_dataset(dataset: str):
    """Parse a single informe_correlacion_qpp.txt and extract ndcg@10 values."""
    path = BASE / dataset / "informe_correlacion_qpp.txt"
    text = path.read_text(encoding="utf-8")

    result = {m: {} for m in METHODS}

    for cor in COR_TYPES:
        m = re.search(
            rf"Correlaciones {cor}:(.*?)(?:\nCorrelaciones |\Z)",
            text,
            flags=re.DOTALL,
        )
        if not m:
            raise RuntimeError(f"No se encontró la sección para {cor} en {dataset}")
        block = m.group(1)

        m_table = re.search(
            r"Valores de correlaci[óo]n[^\n]*\n(.*?)(?:\n\n|Estadísticas Resumen)",
            block,
            flags=re.DOTALL,
        )
        if not m_table:
            raise RuntimeError(f"No se encontró la tabla de valores en {dataset} / {cor}")

        table = m_table.group(1)
        lines = [ln.strip() for ln in table.splitlines() if ln.strip()]
        if not lines:
            continue

        # Primera línea: encabezado "ndcg@10 ap"
        for line in lines[1:]:
            # Ejemplo de línea:
            # "IDF Promedio  0.096422  0.096616"
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

    # Imprime JSON para depuración (solo ASCII para evitar problemas de encoding)
    print("# JSON bruto de correlaciones (ndcg@10):", flush=True)
    print(json.dumps(data, indent=2, ensure_ascii=True), flush=True)

    # También genera filas de Typst listas para pegar en la tabla
    print("\n# Filas Typst sugeridas (ndcg@10, P-rho / S-rho / K-tau):\n", flush=True)

    # Orden lógico para las columnas de correlación en la tabla
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


