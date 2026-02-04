# DataScout
### AI-Powered Dataset Intelligence Platform

**Made for FWD 2026 by Illia Shybanov**

---

DataScout is a data analysis platform that automates exploratory data analysis (EDA). It replaces hours of manual pandas/matplotlib work with a single CSV upload.

Built with **Streamlit** and powered by a local **LLM (Ollama)**, DataScout enables users to instantly profile datasets, detect anomalies, visualize patterns, and chat with their data using natural language.

---

## Key Features

### 1. Instant Data Profiling
- Automatic calculation of summary statistics (rows, duplicates, missing values)
- Deep column analysis: cardinality, data types, unique value counts
- Memory usage overview

### 2. Smart Auto-Visualization
- **Context-Aware Charts**: Automatically selects the best visualization based on column types
- **Distributions**: Histograms and box plots for numeric data
- **Correlations**: Heatmaps to identify relationships between variables
- **Category Breakdowns**: Bar charts for categorical columns

### 3. Intelligent Anomaly Detection
- **Unsupervised Learning**: Uses Isolation Forest to detect multivariate outliers
- **Adaptive Thresholds**: Automatically adjusts sensitivity based on dataset skewness
- **Missing Value Analysis**: Flags rare missing values while ignoring structural nulls
- **Explainability**: Provides human-readable reasons for why a record is anomalous

### 4. AI-Powered Data Chat
- **Local LLM Integration**: Privacy-first analysis using Ollama (Gemma 3 / Mistral)
- **Context-Aware**: The LLM receives dataset statistics, anomalies, and correlations
- **Natural Language**: Ask questions like "Why do we have negative profit?" or "Which columns are correlated?"


---

## Demo Datasets

Download these to test the app:
- [Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — Retail analytics with sales, profit, and regional data
- [Titanic](https://www.kaggle.com/datasets/yasserh/titanic-dataset) — Passenger survival data with mixed column types

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy, Scikit-learn |
| Visualization | Plotly Express |
| AI/LLM | Ollama (local inference) |
| Styling | Custom CSS |

---

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running

### 1. Clone the Repository
```bash
git clone https://github.com/Illia-Shyn/DataScout-AI-Powered-Dataset-Intelligence-Platform.git
cd DataScout-AI-Powered-Dataset-Intelligence-Platform
```

### 2. Set up Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Pull LLM Model
Ensure Ollama is running, then pull the model:
```bash
ollama pull gemma3:4b
```

---

## Usage

1. **Start the App**:
    ```bash
    streamlit run app.py
    ```

2. **Upload Data**: Drag and drop any CSV file into the sidebar.

3. **Explore**:
    - **Profiling** — Dataset health check and column statistics
    - **Patterns** — Auto-generated visualizations
    - **Anomalies** — Flagged records with explanations
    - **AI Chat** — Ask questions about your data

---

## Project Structure

```
DataScout/
├── app.py                  # Main application entry point
├── components/
│   ├── data_profiler.py    # Statistical analysis engine
│   ├── visualizer.py       # Plotly chart generation
│   ├── anomaly_detector.py # Isolation Forest and statistical detection
│   └── llm_chat.py         # Ollama LLM integration
├── utils/
│   └── prompts.py          # LLM prompt templates
├── assets/
│   └── style.css           # Custom styling
├── demo_data/              # Sample datasets
├── requirements.txt
└── README.md
```

---

## Sample Questions for AI Chat

**Superstore Dataset:**
- What is this dataset about?
- Which columns are most correlated?
- What is the average profit by category?
- What is wrong with row 27?

**Titanic Dataset:**
- What factors correlate with survival?
- Which columns have missing values?
- What is the average fare by passenger class?
- Why was row 27 flagged as an anomaly?

---

## Author

**Illia Shybanov**

Created for FWD 2026 Career Fair  
Focus: Full-Stack Data Science and AI Engineering

GitHub: [github.com/Illia-Shyn](https://github.com/Illia-Shyn)

---

## License

MIT License — Free to use and modify.