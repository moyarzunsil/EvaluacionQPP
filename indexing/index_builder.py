import pyterrier as pt
from utils.text_processing import preprocess_text
import json
import os
import shutil
import time

class IndexBuilder:
    def __init__(self, dataset, dataset_name):
        self.dataset = dataset
        self.total_docs = 0
        self.term_df = {}
        self.term_cf = {}
        self.index = None 
        self.total_terms = 0
        self.dataset_name = dataset_name

    def build_index(self, base_index_path):
        os.makedirs(base_index_path, exist_ok=True)
        
        if not pt.started():
            pt.init()

        # Basic indexer configuration (store docno and text in forward meta)
        indexer = pt.IterDictIndexer(
            base_index_path,
            meta={
                "docno": 64,
                "text": 100000
            }
        )
        
        # Track term frequencies during indexing
        term_stats = {}
        
        def doc_iterator():
            iterator = self.dataset.iter_docs()
            for doc_id, doc_text in iterator:
                # Track term frequencies using our preprocessor
                terms = preprocess_text(doc_text, dataset_name=self.dataset_name)
                
                # Track term frequencies
                for term in terms:
                    if term not in term_stats:
                        term_stats[term] = {'df': 0, 'cf': 0}
                    term_stats[term]['cf'] += 1
                    
                # Track document frequencies (unique terms per doc)
                for term in set(terms):
                    term_stats[term]['df'] += 1
                
                self.total_docs += 1
                
                # Pass preprocessed text for indexing
                processed_text = ' '.join(terms)
                yield {
                    'docno': str(doc_id),
                    'text': processed_text,
                }
                # Debugging: Print each document ID and the number of terms
                print(f"Indexed doc_id: {doc_id}, terms count: {len(terms)}")

        # Build index
        index_ref = indexer.index(doc_iterator())
        index = pt.IndexFactory.of(index_ref)
        self.index = index
        
        # Store our tracked statistics
        self.term_df = {term: stats['df'] for term, stats in term_stats.items()}
        self.term_cf = {term: stats['cf'] for term, stats in term_stats.items()}
        self.total_terms = sum(self.term_cf.values())
        # Persist lightweight stats to avoid re-iterating the dataset when loading an existing index
        try:
            self._save_index_stats(base_index_path)
        except Exception as e:
            print(f"Warning: could not persist index stats: {e}")
        
        print("\n=== Index Build Statistics ===")
        print(f"Total documents indexed: {self.total_docs}")
        print(f"Total terms: {self.total_terms}")
        print(f"Unique terms: {len(self.term_df)}")
        print("\nSample term statistics (first 5 terms):")
        for term in list(self.term_df.keys())[:5]:
            print(f"Term '{term}': df={self.term_df[term]}, cf={self.term_cf[term]}")
        
        return index

    def _load_statistics_from_index(self):
        """Load term statistics by processing documents from dataset, similar to build_index"""
        before_total_docs = self.total_docs if hasattr(self, 'total_docs') else 0
        before_total_terms = self.total_terms if hasattr(self, 'total_terms') else 0
        
        # Reset statistics
        self.term_df = {}
        self.term_cf = {}
        self.total_docs = 0
        term_stats = {}
        
        # Process all documents from dataset (same as build_index)
        iterator = self.dataset.iter_docs()
        for doc_id, doc_text in iterator:
            # Process terms using same preprocessor as build_index
            terms = preprocess_text(doc_text, dataset_name=self.dataset_name)
            
            # Track term frequencies (same as in build_index)
            for term in terms:
                if term not in term_stats:
                    term_stats[term] = {'df': 0, 'cf': 0}
                term_stats[term]['cf'] += 1
            
            # Track document frequencies (same as in build_index)
            for term in set(terms):
                term_stats[term]['df'] += 1
            
            self.total_docs += 1

        # Store statistics in the same way as build_index
        self.term_df = {term: stats['df'] for term, stats in term_stats.items()}
        self.term_cf = {term: stats['cf'] for term, stats in term_stats.items()}
        self.total_terms = sum(self.term_cf.values())

        print("\n=== Index Load Statistics ===")
        print(f"Previous total docs: {before_total_docs} -> New: {self.total_docs}")
        print(f"Previous total terms: {before_total_terms} -> New: {self.total_terms}")
        print(f"Unique terms loaded: {len(self.term_df)}")
        print("\nSample term statistics (first 5 terms):")
        for term in list(self.term_df.keys())[:5]:
            print(f"Term '{term}': df={self.term_df[term]}, cf={self.term_cf[term]}")

    def load_or_build_index(self, base_index_path):
        print(f"Checking for index at: {base_index_path}")
        
        if os.path.exists(base_index_path) and os.listdir(base_index_path):
            print(f"Found existing index for dataset {self.dataset_name}. Loading...")
            try:
                indexref = pt.IndexRef.of(base_index_path)
                index = pt.IndexFactory.of(indexref)
                self.index = index
                
                # Load lightweight stats from disk or fall back to Terrier collection stats.
                # Avoid iterating the dataset to prevent re-downloading large corpora.
                loaded_stats = False
                try:
                    loaded_stats = self._load_index_stats(base_index_path)
                except Exception as e:
                    print(f"Warning: could not load persisted stats: {e}")
                
                # If term_cf is empty (either not loaded or stats file was incomplete),
                # populate from the Terrier lexicon
                if not loaded_stats or not self.term_cf:
                    if loaded_stats and not self.term_cf:
                        print("Stats loaded but term_cf is empty, loading from lexicon...")
                    self._populate_stats_from_collection()
                    
                print(f"Successfully loaded existing index for {self.dataset_name}")
                print(f"Total documents: {self.total_docs}")
                print(f"Total terms: {self.total_terms}")
                print(f"Unique terms: {len(self.term_df)}")
                
                
            except Exception as e:
                print(f"Error loading existing index: {e}")
                print("Creating new index...")
                shutil.rmtree(base_index_path)
                index = self.build_index(base_index_path)
        else:
            print(f"No existing index found for dataset {self.dataset_name}. Creating new index...")
            if os.path.exists(base_index_path):
                shutil.rmtree(base_index_path)
            index = self.build_index(base_index_path)
        
        print(f"Term_df entries (sample): {list(self.term_df.items())[:5]}")  # Log term_df sample
        return index

    def _stats_file_path(self, base_index_path):
        return os.path.join(base_index_path, "index_stats.json")

    def _save_index_stats(self, base_index_path, max_terms_to_store=0):
        """
        Save lightweight index stats to disk. For large collections, only totals are stored.
        """
        stats = {
            "dataset": self.dataset_name,
            "total_docs": self.total_docs,
            "total_terms": int(self.total_terms),
            "unique_terms": len(self.term_df),
        }
        # Optionally store a small slice of term stats if requested
        if max_terms_to_store > 0 and self.term_cf:
            sample_terms = list(self.term_cf.keys())[:max_terms_to_store]
            stats["term_cf"] = {t: int(self.term_cf[t]) for t in sample_terms}
            stats["term_df"] = {t: int(self.term_df.get(t, 0)) for t in sample_terms}
        with open(self._stats_file_path(base_index_path), "w", encoding="utf-8") as f:
            json.dump(stats, f)

    def _load_index_stats(self, base_index_path):
        """
        Load persisted index stats if available. Returns True if loaded.
        """
        path = self._stats_file_path(base_index_path)
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        self.total_docs = int(stats.get("total_docs", 0))
        self.total_terms = int(stats.get("total_terms", 0))
        self.term_cf = stats.get("term_cf", {}) or {}
        self.term_df = stats.get("term_df", {}) or {}
        return True

    def _populate_stats_from_collection(self):
        """
        Populate stats from Terrier collection statistics and lexicon (no dataset iteration).
        """
        try:
            cs = self.index.getCollectionStatistics()
            # Try common attribute names across Terrier versions
            for getter in ("getNumberOfTokens", "getNumberOfTokensInCollection"):
                if hasattr(cs, getter):
                    self.total_terms = int(getattr(cs, getter)())
                    break
            # Also populate number of documents
            if hasattr(cs, "getNumberOfDocuments"):
                self.total_docs = int(cs.getNumberOfDocuments())
            elif hasattr(cs, "getNumberOfDocumentsInCollection"):
                self.total_docs = int(cs.getNumberOfDocumentsInCollection())
        except Exception as e:
            print(f"Warning: could not read Terrier collection statistics: {e}")
        
        # Populate term_cf and term_df from the Terrier lexicon
        # This avoids re-iterating the dataset for large corpora
        try:
            lexicon = self.index.getLexicon()
            print(f"Loading term frequencies from Terrier lexicon...")
            term_cf = {}
            term_df = {}
            count = 0
            for entry in lexicon:
                term = entry.getKey()
                # getFrequency() returns collection frequency (cf)
                # getDocumentFrequency() returns document frequency (df)
                term_cf[term] = int(entry.getValue().getFrequency())
                term_df[term] = int(entry.getValue().getDocumentFrequency())
                count += 1
                if count % 50000 == 0:
                    print(f"  Loaded {count} terms...")
            self.term_cf = term_cf
            self.term_df = term_df
            print(f"Loaded {len(self.term_cf)} terms from lexicon")
        except Exception as e:
            print(f"Warning: could not read term frequencies from lexicon: {e}")

    def get_document_terms(self, doc_id):
        """
        Retrieve and preprocess terms from a document given its doc_id.
        """
        doc = self.index.getStore().getDocument(doc_id)
        text = doc['text']
        return preprocess_text(text)

    def get_vocabulary(self):
        """
        Retrieve the complete vocabulary from the collection.
        """
        lexicon = self.index.getLexicon()
        return [entry.getKey() for entry in lexicon]

    def get_term_probability(self, term):
        """
        Calculate P(w|D) for a term based on collection frequencies.
        """
        term_cf = self.term_cf.get(term, 0)
        return term_cf / self.total_terms if self.total_terms > 0 else 0.0

    def save_sample_frequencies_to_json(self, base_file_path, num_samples=100):
        """
        Save a sample of term_df and term_cf to a JSON file with dataset-specific naming.
        """
        # Create dataset-specific filename
        filename = f"sample_term_frequencies_{self.dataset_name}.json"
        file_path = os.path.join(os.path.dirname(base_file_path), filename)
        
        sample_terms = list(self.term_df.keys())[:num_samples]
        sample_data = {
            "dataset": self.dataset_name,
            "term_df": {term: self.term_df[term] for term in sample_terms},
            "term_cf": {term: self.term_cf[term] for term in sample_terms}
        }
        with open(file_path, 'w') as f:
            json.dump(sample_data, f, indent=4)
        print(f"Sample term frequencies for dataset {self.dataset_name} saved to {file_path}")
