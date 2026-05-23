import pyterrier as pt
import pandas as pd
from .iquique_dataset import IquiqueDataset
from utils.config import AVAILABLE_DATASETS, DATASET_FORMATS
from utils.text_processing import preprocess_text
import warnings

class DatasetProcessor:
    def __init__(self, dataset_name: str):
        """
        Initialize the dataset processor with the specified dataset.
        
        Args:
            dataset_name (str): Name/identifier of the dataset to process
        """
        # Keep original path for config lookup
        self.dataset_path = dataset_name
        if dataset_name == "iquique_dataset":
            self.dataset = IquiqueDataset()
        elif dataset_name.startswith("irds:"):
            if not pt.started():
                pt.init()
            self.dataset = pt.get_dataset(dataset_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_queries(self):
        """
        Get both raw and preprocessed queries from the dataset.
        
        Returns:
            dict: Dictionary mapping query IDs to query information containing:
                - 'raw': original query text
                - 'processed': list of preprocessed tokens
        """
        topics = self.dataset.get_topics()
        queries = {}
        
        if isinstance(topics, pd.DataFrame):
            # Compose 'query' column if missing (e.g., CAR has title/headings/text)
            if 'query' not in topics.columns:
                # Resolve dataset key for strategy
                dataset_key = next((k for k, v in AVAILABLE_DATASETS.items() if v == self.dataset_path), None)
                strategy = DATASET_FORMATS.get(dataset_key, {}).get('query_field_strategy', None)
                # Default to title+headings for CAR-like datasets
                if strategy is None and any(c in topics.columns for c in ['title', 'headings', 'text']):
                    strategy = 'title+headings'
                if strategy:
                    def to_str(x):
                        if isinstance(x, (list, tuple)):
                            return ' '.join([str(t) for t in x if pd.notna(t)])
                        return '' if pd.isna(x) else str(x)
                    title_col = topics['title'] if 'title' in topics.columns else pd.Series([''] * len(topics))
                    headings_col = topics['headings'] if 'headings' in topics.columns else pd.Series([''] * len(topics))
                    text_col = topics['text'] if 'text' in topics.columns else pd.Series([''] * len(topics))
                    # Build composed query
                    if strategy == 'title+headings':
                        topics['query'] = title_col.fillna('').astype(str) + ' ' + headings_col.apply(to_str)
                    elif strategy == 'title':
                        topics['query'] = title_col.fillna('').astype(str)
                    elif strategy == 'text':
                        topics['query'] = text_col.fillna('').astype(str)
                    else:
                        topics['query'] = title_col.fillna('').astype(str) + ' ' + headings_col.apply(to_str)
                    # One-line diagnostic print with avg lengths
                    try:
                        avg_title = float(title_col.fillna('').astype(str).str.len().mean()) if 'title' in topics.columns else 0.0
                        avg_head = float(headings_col.apply(to_str).str.len().mean()) if 'headings' in topics.columns else 0.0
                        avg_text = float(text_col.fillna('').astype(str).str.len().mean()) if 'text' in topics.columns else 0.0
                        print(f"CAR queries: strategy={strategy} | avg_len title={avg_title:.1f}, headings={avg_head:.1f}, text={avg_text:.1f}")
                    except Exception:
                        pass
                else:
                    raise ValueError("Topics DataFrame lacks 'query' and no strategy to compose it.")
            raw_queries = dict(zip(topics['qid'].astype(str), topics['query']))
        elif isinstance(topics, dict):
            raw_queries = topics
        elif isinstance(topics, list):
            if all(isinstance(topic, dict) for topic in topics):
                raw_queries = {topic.get('qid', topic.get('query_id')): topic.get('query', topic.get('text')) 
                             for topic in topics}
            elif all(hasattr(topic, 'query_id') and hasattr(topic, 'text') for topic in topics):
                raw_queries = {topic.query_id: topic.text for topic in topics}
            else:
                raise ValueError("Unsupported topic format")
        else:
            raise ValueError("Unsupported topic format")

        # Determine dataset language
        dataset_name = getattr(self.dataset, 'name', '')
        
        # Preprocess each query
        for qid, query_text in raw_queries.items():
            queries[qid] = {
                'raw': query_text,  # Keep original query text
                'processed': preprocess_text(query_text, dataset_name)  # Add preprocessed tokens
            }
        
        return queries

    def get_raw_queries(self):
        """Get original unprocessed queries for retrieval."""
        queries = self.get_queries()
        # For CAR datasets, avoid TerrierQL parsing issues (e.g., '/' in headings)
        # by using preprocessed tokens as the retrieval query string.
        dataset_path_str = str(self.dataset_path)
        if dataset_path_str.startswith("irds:car/") or "car/v1.5" in dataset_path_str:
            raw_map = {qid: ' '.join(query['processed']) for qid, query in queries.items()}
            # Single diagnostic line
            try:
                lengths = [len(v) for v in raw_map.values()]
                avg_len = sum(lengths) / max(1, len(lengths))
                print(f"CAR retrieval: using preprocessed tokens as query (avg_len={avg_len:.1f})")
            except Exception:
                pass
            return raw_map
        return {qid: query['raw'] for qid, query in queries.items()}

    def get_processed_queries(self):
        """Get preprocessed query tokens for QPP methods."""
        queries = self.get_queries()
        return {qid: query['processed'] for qid, query in queries.items()}

    def get_qrels(self):
        return self.dataset.get_qrels()

    def iter_docs(self):
        for doc in self.dataset.get_corpus_iter():
            text = doc['text']
            if isinstance(text, bytes):
                try:
                    text = text.decode('utf-8')
                except UnicodeDecodeError:
                    warnings.warn(f"UTF-8 decode failed for document {doc['docno']}. Falling back to Latin-1.")
                    try:
                        text = text.decode('latin-1')
                    except UnicodeDecodeError:
                        warnings.warn(f"Latin-1 decode failed for document {doc['docno']}. Ignoring problematic characters.")
                        text = text.decode('utf-8', errors='ignore')
            
            yield doc['docno'], text