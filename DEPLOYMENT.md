# Deployment quick start

## Streamlit Community Cloud

1. Push this project directory to GitHub so `streamlit_app.py`, `requirements.txt`,
   `App/`, and `data/documents/` are in the same repository directory.
2. Create the app in Streamlit Community Cloud with entrypoint:
   `streamlit_app.py`.
3. In **Advanced settings -> Secrets**, add:

```toml
GROQ_API_KEY = "your_key_here"
```

4. Select Python 3.12 (the current Community Cloud default) and deploy.
5. If the build fails, open the app's logs and check the first Python/package
   error; dependency changes are picked up after committing `requirements.txt`.

The chat uses Groq `openai/gpt-oss-20b`. The old `llama-3.1-8b-instant` model
was deprecated by Groq on August 16, 2026.

## Local

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env  # Windows
# or: cp .env.example .env
streamlit run streamlit_app.py
```

Never commit `.env` or real patient data.
