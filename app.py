from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pickle
import numpy as np

# Create Flask app
app = Flask(__name__)
# Secret key for flash messages (in production, keep this secret and use env vars)
app.secret_key = 'replace-this-with-a-secure-random-key'

# Paths to the pre-trained model and vectorizer (assumed in project root)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')


def load_pickle(path):
    """Load a pickle file and return the object.

    Raises FileNotFoundError if the file isn't present.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


# Load model and vectorizer at startup
try:
    model = load_pickle(MODEL_PATH)
    vectorizer = load_pickle(VECTORIZER_PATH)
except Exception as e:
    # If loading fails, keep placeholders and handle at request time
    model = None
    vectorizer = None
    load_error = str(e)
else:
    load_error = None


@app.route('/', methods=['GET'])
def index():
    """Render the homepage with optional load error message."""
    # Check for any prediction stored in the session (set by PRG flow)
    prediction = session.pop('prediction', None)
    input_text = session.pop('input_text', None)
    return render_template('index.html', load_error=load_error, prediction=prediction, input_text=input_text)


@app.route('/predict', methods=['POST'])
def predict():
    """Handle text input or uploaded .txt file, validate, vectorize, predict, and render result."""
    global model, vectorizer

    # Re-check model availability
    if model is None or vectorizer is None:
        flash('Model or vectorizer not loaded. Check server logs.', 'danger')
        return redirect(url_for('index'))

    # Get text from textarea
    text_input = request.form.get('text_input', '')

    # Check for uploaded file
    uploaded_file = request.files.get('text_file')

    content = ''

    if uploaded_file and uploaded_file.filename != '':
        # Validate file extension
        filename = uploaded_file.filename
        if not filename.lower().endswith('.txt'):
            flash('Only .txt files are accepted for upload.', 'warning')
            return redirect(url_for('index'))

        try:
            file_bytes = uploaded_file.read()
            # Attempt to decode as utf-8, fallback to latin-1
            try:
                content = file_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = file_bytes.decode('latin-1')
        except Exception:
            flash('Could not read the uploaded file. Make sure it is a valid text file.', 'danger')
            return redirect(url_for('index'))
    else:
        content = text_input.strip()

    # Validate content
    if not content or content.strip() == '':
        flash('Please provide text input or upload a non-empty .txt file.', 'warning')
        return redirect(url_for('index'))

    # Make prediction
    try:
        X = vectorizer.transform([content])
        pred = model.predict(X)
        # Handle different model output shapes/types
        if isinstance(pred, (list, tuple, np.ndarray)):
            label = pred[0]
        else:
            label = pred

        # Map numeric labels to human-readable form if necessary
        if isinstance(label, (int, np.integer)):
            result = 'Positive' if int(label) == 1 else 'Negative'
        elif isinstance(label, str):
            # If model returns 'pos'/'neg' or 'positive'/'negative'
            l = label.lower()
            if l in ('positive', 'pos', '1'):
                result = 'Positive'
            else:
                result = 'Negative'
        else:
            result = str(label)

        # Use Post-Redirect-Get: store prediction in session then redirect to index
        # This prevents the form re-submission issue and ensures a fresh page load for the next input
        session['prediction'] = result
        # store a truncated version of the input for re-display (avoid storing huge data in session)
        session['input_text'] = (content[:1000] + '...') if len(content) > 1000 else content
        return redirect(url_for('index'))

    except Exception as e:
        # Log the exception server-side and show friendly message
        app.logger.exception('Prediction failed')
        flash('An error occurred during prediction. Check the server logs for details.', 'danger')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # When running directly, enable debug mode for development
    app.run(host='0.0.0.0', port=5000, debug=True)
