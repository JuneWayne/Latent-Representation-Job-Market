# Latent Representation of DS Job Market

> **Transformer-Based Semantic Clustering for Discovering Latent Structure in Data Science Job Market Using UMAP and HDBSCAN**

## Interactive Visualization

Explore the full interactive map of the data science job market landscape:

**[View Interactive Visualization](https://raw.githack.com/JuneWayne/Latent-Representation-Job-Market/main/misc/data_science_job_market_landscape.html)**

<p align="center">
  <a href="https://raw.githack.com/JuneWayne/Latent-Representation-Job-Market/main/misc/data_science_job_market_landscape.html">
    <img src="https://img.shields.io/badge/View-Interactive_Map-blue?style=for-the-badge&logo=plotly" alt="Interactive Map"/>
  </a>
</p>

---

## Project Overview

This project implements a transformer-based semantic encoding pipeline coupled with manifold learning and density-based clustering to discover latent structure in data science job descriptions. We leverage **SentenceTransformer** (`all-MiniLM-L6-v2`), a distilled bi-encoder architecture fine-tuned on 1B+ sentence pairs, to map textual job descriptions into a 384-dimensional dense vector space where semantic similarity is preserved via cosine distance.

To address the curse of dimensionality for clustering, we apply **UMAP** (Uniform Manifold Approximation and Projection) for nonlinear dimensionality reduction from 384D to 5D, optimizing for local neighborhood preservation with `n_neighbors=100` and `min_dist=0` to enable tight cluster formation while maintaining topological structure. Clustering is performed using **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise), a density-based algorithm that automatically determines optimal cluster count through persistent homology analysis, configured with `min_cluster_size=100` and `cluster_selection_method='leaf'` for balanced partition discovery.

For interpretability, we extract discriminative features via **class-based TF-IDF**, treating each cluster as a meta-document to identify statistically significant n-grams (unigrams and bigrams). Cluster labeling is automated through prompt engineering with **GPT-4o-mini**, which performs semantic synthesis of top-weighted keywords to generate human-interpretable cluster names grounded in domain knowledge.

### Results

Through this unsupervised learning pipeline on **N=900+ data science job postings**, we identified **k=24 statistically significant clusters** (α=0.05 density threshold) representing distinct functional specializations in the market, including:
- Machine Learning & NLP Engineering
- Business Intelligence & Analytics
- Research Science & Data Engineering
- Industry-specific roles (Healthcare, Finance, Marketing, etc.)

---

## Methodology

### Pipeline Architecture

```
Raw Text (N=900) → Tokenization → SentenceTransformer Encoding (R^384) → UMAP Projection (R^5) → HDBSCAN Clustering (k=24) → Class-based TF-IDF (top-10 n-grams) → GPT-4o-mini Labeling → Plotly Visualization (R^2)
```

### Technical Stack

| Step                            | Technology                               | Mathematical Operation                                             |
| ------------------------------- | ---------------------------------------- | ------------------------------------------------------------------ |
| **1. Data Collection**          | Supabase + LinkedIn API                  | Query N=900 job postings from PostgreSQL database                  |
| **2. Text Embeddings**          | `SentenceTransformer` (all-MiniLM-L6-v2) | f: Text → R^384 via BERT tokenization + mean pooling + L2 norm     |
| **3. Dimensionality Reduction** | `UMAP`                                   | Manifold projection R^384 → R^5 via fuzzy simplicial set theory    |
| **4. Clustering**               | `HDBSCAN`                                | Density-based hierarchical clustering via mutual reachability      |
| **5. Keyword Extraction**       | `TF-IDF` (class-based)                   | Compute log(M/df) × (tf/Σtf) for n-grams per cluster meta-document |
| **6. Semantic Labeling**        | `GPT-4o-mini`                            | Zero-shot classification via prompt engineering (T=0.3)            |
| **7. Visualization**            | `Plotly` + `Matplotlib`                  | Secondary UMAP R^384 → R^2 for web-based interactive scatter       |

---

## Key Features

- **Semantic Vector Space**: Transformer-based embeddings capture distributional semantics beyond lexical matching, enabling cosine-distance similarity computation
- **Automatic Cluster Discovery**: Density-based hierarchical clustering eliminates a priori k specification through topological data analysis
- **Interpretable Feature Extraction**: Class-based TF-IDF identifies statistically discriminative n-grams per cluster (high within-class frequency, low cross-class frequency)
- **Interactive 2D Projection**: Secondary UMAP projection to R^2 for visualization while preserving approximate manifold structure
- **Robust Noise Handling**: HDBSCAN outlier detection identifies low-density anomalies without forcing assignment to nearest cluster

---

## Clustering Results

HDBSCAN identified **k=24 stable density-connected clusters** from N=900 job descriptions, with approximately 15% classified as noise (outliers in low-density regions). Representative clusters include:
- **Machine Learning Engineer**: High-dimensional model development, deployment pipelines, MLOps
- **Data Analytics Engineer**: ETL optimization, analytics infrastructure, cross-functional analytics
- **Business Intelligence Analyst**: Dashboard development, SQL-based reporting, stakeholder communication
- **Healthcare Data Scientist**: Clinical data analysis, regulatory compliance, patient outcome modeling
- **Marketing Analytics**: Attribution modeling, customer segmentation, campaign performance analysis

Each cluster exhibits distinct distributional characteristics in TF-IDF feature space, with inter-cluster cosine similarity < 0.4 and intra-cluster similarity > 0.7, indicating strong semantic separation.

---

## Quick Start

### Prerequisites
```bash
pip install supabase pandas sentence-transformers umap-learn hdbscan scikit-learn plotly openai
```

### Run the Analysis
Open and execute [`Latent-Mapping-Experiment.ipynb`](Latent-Mapping-Experiment.ipynb) to:
1. Load and preprocess job data
2. Generate semantic embeddings
3. Perform clustering analysis
4. Generate interactive visualizations

---

## Project Structure

```
Latent-Representation-Job-Market/
├── Latent-Mapping-Experiment.ipynb    # Main analysis notebook
├── README.md                           # This file
└── misc/
    └── data_science_job_market_landscape.html   # Interactive visualization
```

---

## Technical Highlights

### Architectural Design Decisions

1. **all-MiniLM-L6-v2 Sentence Encoder**: Selected for its optimal balance between computational efficiency (6-layer distilled BERT) and semantic representation quality (trained on 1B+ sentence pairs via contrastive learning). Produces L2-normalized embeddings in R^384 where cosine similarity directly measures semantic relatedness.

2. **UMAP Manifold Learning**: Employs Riemannian geometry and fuzzy topological representation to preserve both local and global structure during dimensionality reduction. Configured with `n_neighbors=100` (broader neighborhood context) and `min_dist=0` (allows tight cluster packing) to optimize downstream clustering separability. Superior to t-SNE for preserving global structure and linear scalability O(N log N).

3. **HDBSCAN Density-Based Clustering**: Utilizes persistent homology to construct a hierarchy of density-based clusters, automatically determining k through stability analysis. Eliminates the need for a priori cluster count specification (unlike k-means) and naturally handles noise points (-1 label). `min_cluster_size=100` ensures statistical robustness, while `cluster_selection_method='leaf'` selects stable leaf clusters for interpretability.

4. **Class-Based TF-IDF Feature Extraction**: Modified TF-IDF formulation where each cluster C_i is treated as a meta-document D_i = concatenate(descriptions ∈ C_i). This approach amplifies cluster-discriminating terms while suppressing globally common vocabulary, enabling better semantic differentiation than traditional per-document TF-IDF.

5. **LLM-Augmented Semantic Labeling**: Zero-shot prompt engineering with GPT-4o-mini to synthesize cluster labels from top-k weighted n-grams. Temperature set to 0.3 for deterministic outputs. This bypasses manual annotation while leveraging the model's implicit domain knowledge for contextually appropriate naming.

### Optimization Hyperparameters

| Parameter                | Value  | Justification                                                |
| ------------------------ | ------ | ------------------------------------------------------------ |
| Embedding Dimension      | 384    | SentenceTransformer output dimensionality                    |
| UMAP Components          | 5      | Balances information preservation with clustering efficiency |
| UMAP Neighbors           | 100    | Captures mesoscale structure in semantic space               |
| UMAP Min Distance        | 0.0    | Enables tight cluster formation for density-based methods    |
| HDBSCAN Min Cluster Size | 100    | n > 100 ensures statistical significance (α=0.05)            |
| HDBSCAN Min Samples      | 2      | Controls core point density threshold                        |
| TF-IDF N-gram Range      | (1, 2) | Captures unigrams and bigrams for interpretability           |
| Top Keywords per Cluster | 10     | Sufficient for LLM context without noise                     |
| GPT Temperature          | 0.3    | Low temperature for consistent, focused outputs              |

---

## Insights & Applications

### For Job Seekers
- Understand specialized niches within data science
- Identify skill gaps for target roles
- Discover emerging sub-fields

### For Recruiters
- Benchmark job descriptions against market clusters
- Identify unique positioning opportunities
- Understand competitive landscape

### For Researchers
- Methodology template for text clustering at scale
- Demonstration of transformer + UMAP + HDBSCAN pipeline
- Example of LLM-augmented unsupervised learning

---

## Acknowledgments

- **Active Learning Lab (UVA):** Dr.Brian Wright, Ali Rivera, Leah Kim

---

<p align="center">
  <i>Mapping the invisible structure of the data science job market</i>
</p>
