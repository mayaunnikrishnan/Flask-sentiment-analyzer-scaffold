# Sentiment Analyzer (Flask)

This is a small Flask web application that uses an existing trained Logistic Regression model and a TF-IDF vectorizer (pickled) to predict sentiment (Positive/Negative).

Files expected in the project root:
- `sentiment_model.pkl`  (your trained model)
- `tfidf_vectorizer.pkl` (your TF-IDF vectorizer)

Quick start (Windows PowerShell):

1. Create a virtual environment

```powershell
python -m venv .venv
```

2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. Run the app

```powershell
python app.py
```

5. Open your browser at http://127.0.0.1:5000

Notes:
- Keep your `sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as `app.py`.
- For production deployment, set a secure `SECRET_KEY` (don't hardcode), disable debug mode, and use a production WSGI server.
