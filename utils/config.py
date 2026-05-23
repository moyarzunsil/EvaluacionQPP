AVAILABLE_DATASETS = {
    "antique_test": "irds:antique/test",
    "iquique_small": "iquique_dataset",
    "cranfield": "irds:cranfield",
    "fiqa": "irds:beir/fiqa/dev",
    "car": "irds:beir/car/dev",
    "msmarco_v2_judged": "irds:msmarco-passage-v2/trec-dl-2022/judged",
    "trec_covid": "irds:beir/trec-covid",
    "msmarco_dl20_judged": "irds:msmarco-passage/trec-dl-2020/judged",
    "car_v15_train_fold0": "irds:car/v1.5/train/fold0",
    "car_v15_trec_y1_manual": "irds:car/v1.5/trec-y1/manual",
    # Add more datasets as needed
    # "msmarco": "irds:msmarco/...",
    # "trec": "irds:trec/...",
}

# Dataset format configurations updated for TREC compliance
DATASET_FORMATS = {
     "antique_test": {
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        # Fixed transformation with type safety
        "doc_id_transform": lambda x: str(x).strip(),
        "relevance_levels": {1: "Out of context", 2: "Not relevant", 3: "Marginal", 4: "Highly relevant"},
        "binary_threshold": 3,
        "gain_values": {1: 0, 2: 1, 3: 2, 4: 3}
    },
    "cranfield": {
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        "relevance_levels": {
            -1: "No interest",
            1: "Minimum interest",
            2: "Useful reference",
            3: "High relevance",
            4: "Complete answer"
        },
        "binary_threshold": 1,  # Consider relevance >= 1 as relevant
        "gain_values": {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4}  # Graduated relevance scale
    },
    "msmarco_v2_judged": {
        # MSMARCO v2 judged uses 0..3 labels; keep mapping flexible via renames
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        # Español: umbral binario en >=1 y ganancias graduadas 0..3
        "relevance_levels": {0: "Non-relevant", 1: "Relevant", 2: "Highly relevant", 3: "Perfect"},
        "binary_threshold": 1,
        "gain_values": {0: 0, 1: 1, 2: 2, 3: 3}
    },
    "trec_covid": {
        # BEIR TREC-COVID (171,332 docs; qrels labels in {-1,0,1,2})
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        # Español: umbral binario >=1 y ganancias 0..2; -1 y 0 mapean a 0
        "relevance_levels": {-1: "Not assessed/negative", 0: "Non-relevant", 1: "Relevant", 2: "Highly relevant"},
        "binary_threshold": 1,
        "gain_values": {-1: 0, 0: 0, 1: 1, 2: 2}
    },
    "msmarco_dl20_judged": {
        # MSMARCO passage TREC-DL 2020 judged (0..3): 0=Irrelevant, 1=Related, 2=Highly, 3=Perfect
        # Español: Para métricas binarias, el umbral oficial suele ser >=2 (RR rel=2)
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        "relevance_levels": {0: "Irrelevant", 1: "Related", 2: "Highly relevant", 3: "Perfectly relevant"},
        "binary_threshold": 2,
        "gain_values": {0: 0, 1: 1, 2: 2, 3: 3}
    },
    "car_v15_train_fold0": {
        # TREC CAR v1.5 train fold 0: qrels solo con nivel 1 (presencia bajo el encabezado)
        # Español: tratamos como binario (>=1 relevante) y ganancias {0,1}
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        "query_field_strategy": "title+headings",
        "relevance_levels": {0: "Non-relevant", 1: "Paragraph under heading"},
        "binary_threshold": 1,
        "gain_values": {0: 0, 1: 1}
    },
    "car_v15_trec_y1_manual": {
        # TREC CAR v1.5 Y1 manual: qrels en {-2,-1,0,1,2,3}
        # Español: para binario usamos >=1 como relevante; ganancias graduadas 0..3
        "qrels_columns": {'qid': 'query_id', 'docno': 'doc_id', 'label': 'relevance'},
        "run_columns": {'qid': 'query_id', 'docno': 'doc_id', 'docScore': 'score'},
        "doc_id_transform": lambda x: str(x).strip(),
        "query_field_strategy": "title+headings",
        "relevance_levels": {
            -2: "Trash",
            -1: "NO, non-relevant",
            0: "Non-relevant, roughly on topic",
            1: "CAN be mentioned",
            2: "SHOULD be mentioned",
            3: "MUST be mentioned"
        },
        "binary_threshold": 1,
        "gain_values": {-2: 0, -1: 0, 0: 0, 1: 1, 2: 2, 3: 3}
    },
}