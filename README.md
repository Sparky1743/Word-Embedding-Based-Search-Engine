# Word Embedding-Based Search Engine

**Team IDS Group (ID: 30)**  
Birudugadda Srivibhav (22110050), Eshwar Sai Ganesh (22110123), Pavan Deekshith Doddi (22110190), Vubbani Bharath Chandra (22110293)

---

## 📌 Project Overview

The goal of this project is to build a semantic search engine that leverages word embeddings to retrieve relevant documents for a given query. We explore various approximate nearest neighbor (ANN) and hashing-based search techniques to balance retrieval accuracy with computational efficiency.

---

## 🎯 Objectives

- Develop a semantic search engine using word embeddings.
- Implement and compare retrieval techniques: Flat (brute-force), HNSW (ANN), MinHash, and SimHash.
- Evaluate performance using information retrieval metrics like Precision@K, Recall@K, MRR, and NDCG.
- Optimize for speed and scalability across large datasets.
- Analyze trade-offs between retrieval quality and performance.

---

## 📂 Dataset

We use the **AG News Corpus** ([AG News CSV](https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz)), consisting of approximately **127,000 news articles** across four categories: World, Sports, Business, and Sci/Tech.

---

## 🧠 Embeddings

All documents and queries are embedded using the **`all-MiniLM-L6-v2`** model from **Sentence Transformers**. This model provides a compact, efficient, and high-quality representation of textual data, making it suitable for large-scale semantic search.

---

## 🔧 Methods Implemented

| Method     | Description                                                                 |
|------------|-----------------------------------------------------------------------------|
| **Flat**   | Brute-force cosine similarity computation (baseline, most accurate)         |
| **HNSW**   | Hierarchical Navigable Small World Graphs for fast ANN retrieval            |
| **MinHash**| Jaccard similarity approximation using token sets (hashing-based method)    |
| **SimHash**| Bitwise similarity hashing for approximate duplicate detection               |

---

## 📊 Evaluation Metrics

We used the following metrics for a thorough evaluation:

- **Precision@K (P@K)**: Proportion of relevant documents in top-K results.
- **MRR (Mean Reciprocal Rank)**: Average reciprocal rank of the first relevant document.
- **NDCG@K**: Normalized Discounted Cumulative Gain emphasizing the order of relevance.
- **Time (s)**: Average retrieval time per query.
- **Speedup (x)**: Relative speedup compared to the slowest method (MinHash/SimHash).

---

## 📈 Results Summary

The following table presents performance metrics averaged across a diverse set of 8 test queries. Each method was evaluated using key retrieval metrics such as **MRR**, **Precision@k**, and **NDCG@k**, as well as runtime and speedup.

| Method   | Time (s)         | Speedup (x) | MRR            | P@1  | NDCG@1 | P@3  | NDCG@3 | P@5  | NDCG@5 | P@10 | NDCG@10 |
|----------|------------------|-------------|----------------|------|--------|------|--------|------|--------|------|----------|
| Flat     | 0.0002 ± 0.0000  | 6598.3      | 0.917 ± 0.220  | 0.875| 0.875  | 0.792| 0.809  | 0.825| 0.827  | 0.837| 0.835    |
| HNSW     | 0.0001 ± 0.0000  | 19289.1     | 0.854 ± 0.256  | 0.75 | 0.75   | 0.792| 0.787  | 0.8  | 0.793  | 0.787| 0.786    |
| MinHash  | 1.2147 ± 0.0344  | 1.0         | 0.900 ± 0.265  | 0.875| 0.875  | 0.542| 0.610  | 0.525| 0.579  | 0.613| 0.621    |
| SimHash  | 1.2147 ± 0.0344  | 1.0         | 0.635 ± 0.365  | 0.5  | 0.5    | 0.333| 0.360  | 0.425| 0.419  | 0.388| 0.393    |

### 🧪 Test Queries Used

The evaluation was performed on the following 8 queries:

```python
TEST_QUERIES = {
    "Impact of oil prices on global economy": "Business",
    "Latest advancements in space exploration": "Sci/Tech",
    "NBA basketball finals highlights": "Sports",
    "Political tensions in the Middle East": "World",
    "New treatments for cancer discovered": "Sci/Tech",
    "Results of recent soccer world cup": "Sports", 
    "Startup company funding rounds": "Business",
    "International summit on climate change": "World"
}
```
---

## 💡 Key Insights

- **Flat search** provides the best accuracy but is slower at scale.
- **HNSW** achieves a good balance between speed and accuracy, with the highest speedup.
- **MinHash** performs reasonably well on P@1 but suffers beyond top-1 ranks.
- **SimHash** is fast but less effective for semantic similarity tasks.

---

## 🌐 Visualizing ANN Graphs (HNSW & Flat) with Feder

You can explore the structure of the ANN graphs (e.g., HNSW, Flat) using the Feder visualizer.

### 📁 Directory Structure

Organize your files as follows:

```
website 
│   index.html
│
├───data
│       agnews_flat.index
│       agnews_hnsw.index
│
└───visualize
        final.html
```

- `agnews_flat.index`, `agnews_hnsw.index`: index files
- `final.html`: Feder visualizer page

### 🚀 Running the Visualizer

1. Navigate to the `website` folder:
```bash
cd website
```

2. Start a simple web server:
```bash
python -m http.server 5000 --bind 127.0.0.1
```

3. Open the following URL in your browser:
```
http://127.0.0.1:5000/
```

### 🧠 Why Visualize?

- Understand the structure of search indices (e.g., shortcut links in HNSW).
- Spot outliers or clusters in your data.
- Explain model behavior visually.

---
## 🔭 Future Work

- Explore larger transformer-based embedding models like `all-mpnet-base-v2` or BERT variants.
- Incorporate query expansion and reranking modules.
- Extend to multi-modal retrieval (e.g., image + text).

---

## 📚 References

- Mikolov, T., et al. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
- Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs*. IEEE T-BD.
- Indyk, P., & Motwani, R. (1998). *Approximate Nearest Neighbors*. STOC.
- Guo, J., et al. (2016). *A Deep Relevance Matching Model for Ad-Hoc Retrieval*. CIKM.

---

## 🛠️ Tools and Libraries

- Python, NumPy, scikit-learn
- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [datasketch](https://ekzhu.github.io/datasketch/)
