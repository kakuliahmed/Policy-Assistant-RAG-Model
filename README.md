## Policy Assistant RAG Model

Lightweight Streamlit app that answers questions about Life Insurance Corporation (LIC) products using a retrieval-augmented generation (RAG) workflow. It queries a local Chroma vector store that indexes `LIC.csv`, then formats the highest-confidence match through the Mistral Small chat model.

### Repo Layout
- `app.py` – Streamlit UI, Chroma lookup, and Mistral prompt formatting.
- `lic_vector_db_v2/` – persisted Chroma collection (`lic_policies`) and metadata files.
- `LIC.csv` – source table for policies (used to build the embeddings).
- `dataset.ipynb` – notebook for cleaning the CSV and populating the vector DB.
- `requirements.txt` – Python dependencies.

### Setup
1. Create and activate a Python 3.10+ virtual environment.
2. `pip install -r requirements.txt`
3. Set `MISTRAL_API_KEY` in your environment (or update `app.py` to load it securely).
4. Ensure `lic_vector_db_v2` is present; regenerate via the notebook if you need fresh embeddings.

### Run
```
streamlit run app.py
```

### Notes
- Retrieval confidence is enforced via a distance threshold of `0.40`; low-confidence queries politely decline.
- The app assumes the top retrieved policy contains the fields `Policy Name`, `Policy ID`, `Description`, and `Duration`. Update the prompt if your metadata schema changes.

