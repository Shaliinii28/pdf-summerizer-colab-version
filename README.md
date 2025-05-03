# PDF Summarizer in Google Colab

This project is a Flask-based web application that summarizes PDF documents using the `sshleifer/distilbart-cnn-6-6` model from Hugging Face. It extracts text from PDFs with `pdfplumber`, supports OCR for scanned PDFs with `pytesseract`, and runs on Google Colab with GPU acceleration. The app is exposed publicly via `ngrok` for testing.

## Prerequisites

- A Google account to access [Google Colab](https://colab.research.google.com).
- A free [ngrok](https://ngrok.com) account to get an authtoken.
- A PDF file (<10MB, text-based or scanned) for testing.

## Setup Instructions

Follow these steps to run the PDF Summarizer in Google Colab.

### 1. Create a New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **New Notebook** to create a blank notebook.

### 2. Enable GPU Runtime
1. In the Colab notebook, go to **Edit > Notebook settings**.
2. Set **Hardware accelerator** to **GPU** (e.g., T4).
3. Click **Save**.

### 3. Get ngrok Authtoken
1. Sign up for a free account at [ngrok.com](https://ngrok.com).
2. Get your authtoken from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).
3. Note the authtoken (e.g., `2abc123...`) for use in the code.

### 4. Copy the Code
The application code is provided in a single script (or split into multiple cells for easier debugging). Choose one approach:

#### Option A: Run as a Single Cell
1. Copy the entire code from the provided notebook script (below or from the source).
2. Paste it into a single code cell in your Colab notebook.
3. Replace `YOUR_NGROK_AUTHTOKEN` in the `!ngrok authtoken` line with your actual ngrok authtoken.

**Single Cell Code**:
```python
# Install dependencies
!pip install flask pyngrok pdfplumber transformers torch pytesseract pillow gunicorn

# Install Tesseract OCR
!apt-get update
!apt-get install -y tesseract-ocr

# Download ngrok
!wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
!tar -xvzf ngrok-v3-stable-linux-amd64.tgz
!mv ngrok /usr/local/bin/

# Set up ngrok authtoken (replace with your ngrok authtoken)
!ngrok authtoken YOUR_NGROK_AUTHTOKEN  # Get this from https://dashboard.ngrok.com/get-started/your-authtoken

# Import libraries
import os
from pyngrok import ngrok
import time

# Create templates directory
os.makedirs('/content/templates', exist_ok=True)

# Write app.py
app_py_content = '''
import os
import re
import io
import hashlib
import tempfile
import logging
from difflib import SequenceMatcher
from flask import Flask, Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pdfplumber
import transformers
import torch
from transformers import pipeline

# Optional OCR fallback
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

# Configuration and Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'colab_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Determine GPU usage
use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1
dtype = torch.float16 if use_gpu else torch.float32
logger.info(f"{'Using GPU' if use_gpu else 'Using CPU'} for model inference")

if use_gpu:
    torch.backends.cuda.matmul.allow_tf32 = True

# Load summarization model
try:
    model_path = "sshleifer/distilbart-cnn-6-6"
    summarizer = pipeline(
        "summarization",
        model=model_path,
        device=device,
        torch_dtype=dtype,
        clean_up_tokenization_spaces=True
    )
except Exception as e:
    logger.error(f"Failed to initialize summarizer: {e}")
    summarizer = None
    raise RuntimeError("Summarization model initialization failed.")

# Blueprint for routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/summarize', methods=['POST'])
def summarize_route():
    if not summarizer:
        return jsonify({'error': 'Summarization model failed to load.'}), 500

    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

    filename = secure_filename(file.filename)
    file_content = file.read()
    if not file_content:
        return jsonify({'error': 'Uploaded file is empty'}), 400

    if len(file_content) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'File size exceeds 10MB limit'}), 400

    try:
        max_words = int(request.form.get('max_words', 200))
        if max_words < 50 or max_words > 500:
            return jsonify({'error': 'Summary length must be between 50 and 500 words'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid summary length'}), 400

    text = extract_text_from_pdf(io.BytesIO(file_content))
    if not text:
        return jsonify({'error': 'No text extracted from PDF'}), 400

    summary = generate_summary(text, max_words)
    return jsonify({'summary': summary})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_stream):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_stream.read())
            tmp_path = tmp.name
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text or not page_text.strip() and pytesseract and Image:
                    try:
                        img = page.to_image(resolution=300).original
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.warning(f"OCR failed for page: {e}")
                        page_text = ""
                text += page_text + "\\n" if page_text else ""
        logger.info(f"Extracted text length: {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Deleted temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def summarize_chunks(chunks, target_length):
    if not summarizer:
        logger.error("Summarizer not initialized")
        return []
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_length = len(chunk.split())
            max_length = max(100, min(target_length, chunk_length // 2 + 50))
            min_length = min(30, max_length // 2)
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            logger.info(f"Summarized chunk {i+1} with {len(summary[0]['summary_text'].split())} words")
        except Exception as e:
            logger.warning(f"Failed to summarize chunk {i+1}: {e}")
            summaries.append("")
    return [s for s in summaries if s]

def clean_summary(summary):
    summary = re.sub(r'\\s+', ' ', summary).strip()
    sentences = [s.strip() + ('.' if not s.endswith(('.', '!', '?')) else '') 
                 for s in re.split(r'(?<=[.!?])\\s+', summary) if s.strip()]
    unique_sentences = []
    seen = []
    for s in sentences:
        normalized = re.sub(r'[^a-z0-9]', '', s.lower())
        if not any(SequenceMatcher(None, normalized, seen_s).ratio() > 0.85 for seen_s in seen):
            unique_sentences.append(s)
            seen.append(normalized)
    return ' '.join(unique_sentences).strip()

def generate_summary(text, max_words=200, retries=0):
    if retries > 2:
        logger.warning("Max retries exceeded")
        return "Summary generation incomplete due to processing errors."
    text = text[:100000]
    if not text.strip():
        return "No valid text to summarize."
    logger.info(f"Input text length: {len(text)} characters")
    chunks = chunk_text(text)
    if not chunks:
        logger.error("No chunks created from text")
        return "Unable to generate summary due to text processing error."
    base_chunks = max(3, min(10, max_words // 100))
    target_words_per_chunk = max(50, max_words // base_chunks + 30)
    if len(chunks) > 3:
        mid_idx = len(chunks) // 2
        prioritized_chunks = [chunks[0], chunks[mid_idx], chunks[-1]] + chunks[1:mid_idx][:base_chunks-3]
    else:
        prioritized_chunks = chunks[:base_chunks]
    summaries = summarize_chunks(prioritized_chunks, target_words_per_chunk)
    if not summaries:
        logger.error("No chunks could be summarized")
        return "Unable to generate summary due to processing errors."
    final_summary = clean_summary(" ".join(summaries))
    sentences = re.split(r'(?<=[.!?])\\s+', final_summary)
    final_words = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) <= max_words:
            final_words.extend(words)
            count += len(words)
        else:
            break
    final_summary = " ".join(final_words).strip()
    input_words = set(text.lower().split()[:1000])
    summary_words = set(final_summary.lower().split())
    overlap = len(input_words.intersection(summary_words)) / max(1, len(summary_words))
    if overlap > 0.6 and count < max_words * 0.95 and retries < 2:
        logger.warning("High overlap detected, retrying summarization")
        return generate_summary(text, max_words, retries + 1)
    logger.info(f"Final summary length: {count} words")
    return final_summary if final_summary else "Summary could not be generated."

app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('/content/app.py', 'w') as f:
    f.write(app_py_content)

# Write index.html
index_html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Summarizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin: 10px 0 5px;
            font-weight: bold;
        }
        input[type="file"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
        #summary, #error {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
        }
        #summary {
            background-color: #e7f3fe;
            border: 1px solid #b3d7ff;
        }
        #error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h1>PDF Summarizer</h1>
    <div class="container">
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="pdf">Upload PDF:</label>
            <input type="file" id="pdf" name="pdf" accept=".pdf" required>
            <label for="max_words">Summary Length (50-500 words):</label>
            <input type="number" id="max_words" name="max_words" min="50" max="500" value="200" required>
            <button type="submit">Summarize</button>
        </form>
        <div id="summary" class="hidden"></div>
        <div id="error" class="hidden"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const summaryDiv = document.getElementById('summary');
            const errorDiv = document.getElementById('error');

            summaryDiv.classList.add('hidden');
            errorDiv.classList.add('hidden');

            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (response.ok && result.summary) {
                    summaryDiv.textContent = result.summary;
                    summaryDiv.classList.remove('hidden');
                } else {
                    errorDiv.textContent = result.error || 'An error occurred';
                    errorDiv.classList.remove('hidden');
                }
            } catch (err) {
                errorDiv.textContent = 'Network error: ' + err.message;
                errorDiv.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
'''

with open('/content/templates/index.html', 'w') as f:
    f.write(index_html_content)

# Start ngrok tunnel
public_url = ngrok.connect(5000, bind_tls=True).public_url
print(f"Flask app is running at: {public_url}")

# Run the Flask app
!gunicorn --bind 0.0.0.0:5000 app:app
```

#### Option B: Run as Multiple Cells (Recommended for Debugging)
1. Create four code cells in your Colab notebook.
2. Paste the following sections into separate cells, replacing `YOUR_NGROK_AUTHTOKEN` in Cell 2 with your ngrok authtoken.

**Cell 1: Install Dependencies**
```python
# Install Python dependencies
!pip install flask pyngrok pdfplumber transformers torch pytesseract pillow gunicorn

# Install Tesseract OCR
!apt-get update
!apt-get install -y tesseract-ocr

# Download and install ngrok
!wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
!tar -xvzf ngrok-v3-stable-linux-amd64.tgz
!mv ngrok /usr/local/bin/
```

**Cell 2: Set Up ngrok and Imports**
```python
# Set up ngrok authtoken (replace with your ngrok authtoken)
!ngrok authtoken YOUR_NGROK_AUTHTOKEN  # Get this from https://dashboard.ngrok.com/get-started/your-authtoken

# Import libraries
import os
from pyngrok import ngrok
import time

# Create templates directory
os.makedirs('/content/templates', exist_ok=True)
```

**Cell 3: Write `app.py` and `index.html`**
```python
# Write app.py
app_py_content = '''
import os
import re
import io
import hashlib
import tempfile
import logging
from difflib import SequenceMatcher
from flask import Flask, Blueprint, request, jsonify, render_template
from werkzeug.utils import secure_filename
import pdfplumber
import transformers
import torch
from transformers import pipeline

# Optional OCR fallback
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

# Configuration and Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'colab_secret_key'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB upload limit
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Determine GPU usage
use_gpu = torch.cuda.is_available()
device = 0 if use_gpu else -1
dtype = torch.float16 if use_gpu else torch.float32
logger.info(f"{'Using GPU' if use_gpu else 'Using CPU'} for model inference")

if use_gpu:
    torch.backends.cuda.matmul.allow_tf32 = True

# Load summarization model
try:
    model_path = "sshleifer/distilbart-cnn-6-6"
    summarizer = pipeline(
        "summarization",
        model=model_path,
        device=device,
        torch_dtype=dtype,
        clean_up_tokenization_spaces=True
    )
except Exception as e:
    logger.error(f"Failed to initialize summarizer: {e}")
    summarizer = None
    raise RuntimeError("Summarization model initialization failed.")

# Blueprint for routes
main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/summarize', methods=['POST'])
def summarize_route():
    if not summarizer:
        return jsonify({'error': 'Summarization model failed to load.'}), 500

    if 'pdf' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['pdf']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PDF file.'}), 400

    filename = secure_filename(file.filename)
    file_content = file.read()
    if not file_content:
        return jsonify({'error': 'Uploaded file is empty'}), 400

    if len(file_content) > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({'error': 'File size exceeds 10MB limit'}), 400

    try:
        max_words = int(request.form.get('max_words', 200))
        if max_words < 50 or max_words > 500:
            return jsonify({'error': 'Summary length must be between 50 and 500 words'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid summary length'}), 400

    text = extract_text_from_pdf(io.BytesIO(file_content))
    if not text:
        return jsonify({'error': 'No text extracted from PDF'}), 400

    summary = generate_summary(text, max_words)
    return jsonify({'summary': summary})

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(pdf_stream):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_stream.read())
            tmp_path = tmp.name
        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if not page_text or not page_text.strip() and pytesseract and Image:
                    try:
                        img = page.to_image(resolution=300).original
                        page_text = pytesseract.image_to_string(img)
                    except Exception as e:
                        logger.warning(f"OCR failed for page: {e}")
                        page_text = ""
                text += page_text + "\\n" if page_text else ""
        logger.info(f"Extracted text length: {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.info(f"Deleted temporary file: {tmp_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")

def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def summarize_chunks(chunks, target_length):
    if not summarizer:
        logger.error("Summarizer not initialized")
        return []
    summaries = []
    for i, chunk in enumerate(chunks):
        try:
            chunk_length = len(chunk.split())
            max_length = max(100, min(target_length, chunk_length // 2 + 50))
            min_length = min(30, max_length // 2)
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(summary[0]['summary_text'])
            logger.info(f"Summarized chunk {i+1} with {len(summary[0]['summary_text'].split())} words")
        except Exception as e:
            logger.warning(f"Failed to summarize chunk {i+1}: {e}")
            summaries.append("")
    return [s for s in summaries if s]

def clean_summary(summary):
    summary = re.sub(r'\\s+', ' ', summary).strip()
    sentences = [s.strip() + ('.' if not s.endswith(('.', '!', '?')) else '') 
                 for s in re.split(r'(?<=[.!?])\\s+', summary) if s.strip()]
    unique_sentences = []
    seen = []
    for s in sentences:
        normalized = re.sub(r'[^a-z0-9]', '', s.lower())
        if not any(SequenceMatcher(None, normalized, seen_s).ratio() > 0.85 for seen_s in seen):
            unique_sentences.append(s)
            seen.append(normalized)
    return ' '.join(unique_sentences).strip()

def generate_summary(text, max_words=200, retries=0):
    if retries > 2:
        logger.warning("Max retries exceeded")
        return "Summary generation incomplete due to processing errors."
    text = text[:100000]
    if not text.strip():
        return "No valid text to summarize."
    logger.info(f"Input text length: {len(text)} characters")
    chunks = chunk_text(text)
    if not chunks:
        logger.error("No chunks created from text")
        return "Unable to generate summary due to text processing error."
    base_chunks = max(3, min(10, max_words // 100))
    target_words_per_chunk = max(50, max_words // base_chunks + 30)
    if len(chunks) > 3:
        mid_idx = len(chunks) // 2
        prioritized_chunks = [chunks[0], chunks[mid_idx], chunks[-1]] + chunks[1:mid_idx][:base_chunks-3]
    else:
        prioritized_chunks = chunks[:base_chunks]
    summaries = summarize_chunks(prioritized_chunks, target_words_per_chunk)
    if not summaries:
        logger.error("No chunks could be summarized")
        return "Unable to generate summary due to processing errors."
    final_summary = clean_summary(" ".join(summaries))
    sentences = re.split(r'(?<=[.!?])\\s+', final_summary)
    final_words = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) <= max_words:
            final_words.extend(words)
            count += len(words)
        else:
            break
    final_summary = " ".join(final_words).strip()
    input_words = set(text.lower().split()[:1000])
    summary_words = set(final_summary.lower().split())
    overlap = len(input_words.intersection(summary_words)) / max(1, len(summary_words))
    if overlap > 0.6 and count < max_words * 0.95 and retries < 2:
        logger.warning("High overlap detected, retrying summarization")
        return generate_summary(text, max_words, retries + 1)
    logger.info(f"Final summary length: {count} words")
    return final_summary if final_summary else "Summary could not be generated."

app.register_blueprint(main_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
'''

with open('/content/app.py', 'w') as f:
    f.write(app_py_content)

# Write index.html
index_html_content = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDF Summarizer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        label {
            display: block;
            margin: 10px 0 5px;
            font-weight: bold;
        }
        input[type="file"], input[type="number"] {
            width: 100%;
            padding: 8px;
            margin-bottom: 10px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        button {
            background-color: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        button:hover {
            background-color: #218838;
        }
        #summary, #error {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
        }
        #summary {
            background-color: #e7f3fe;
            border: 1px solid #b3d7ff;
        }
        #error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }
        .hidden {
            display: none;
        }
    </style>
</head>
<body>
    <h1>PDF Summarizer</h1>
    <div class="container">
        <form id="uploadForm" enctype="multipart/form-data">
            <label for="pdf">Upload PDF:</label>
            <input type="file" id="pdf" name="pdf" accept=".pdf" required>
            <label for="max_words">Summary Length (50-500 words):</label>
            <input type="number" id="max_words" name="max_words" min="50" max="500" value="200" required>
            <button type="submit">Summarize</button>
        </form>
        <div id="summary" class="hidden"></div>
        <div id="error" class="hidden"></div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const formData = new FormData(form);
            const summaryDiv = document.getElementById('summary');
            const errorDiv = document.getElementById('error');

            summaryDiv.classList.add('hidden');
            errorDiv.classList.add('hidden');

            try {
                const response = await fetch('/summarize', {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();

                if (response.ok && result.summary) {
                    summaryDiv.textContent = result.summary;
                    summaryDiv.classList.remove('hidden');
                } else {
                    errorDiv.textContent = result.error || 'An error occurred';
                    errorDiv.classList.remove('hidden');
                }
            } catch (err) {
                errorDiv.textContent = 'Network error: ' + err.message;
                errorDiv.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>
'''

with open('/content/templates/index.html', 'w') as f:
    f.write(index_html_content)
```

**Cell 4: Start ngrok and Run Flask App**
```python
# Start ngrok tunnel
public_url = ngrok.connect(5000, bind_tls=True).public_url
print(f"Flask app is running at: {public_url}")

# Run the Flask app
!gunicorn --bind 0.0.0.0:5000 app:app
```

3. Save the notebook (e.g., `pdf_summarizer.ipynb`) to your Google Drive for reuse.

### 5. Run the Code
1. **For Option A (Single Cell)**:
   - Paste the single cell code into a code cell.
   - Replace `YOUR_NGROK_AUTHTOKEN` with your ngrok authtoken.
   - Run the cell (Shift + Enter).
   - Wait for dependencies to install, files to be written, and the app to start (may take 5–10 minutes).

2. **For Option B (Multiple Cells)**:
   - Paste each cell’s code into a separate code cell in the notebook.
   - Replace `YOUR_NGROK_AUTHTOKEN` in Cell 2 with your ngrok authtoken.
   - Run the cells in order (Shift + Enter for each):
     - **Cell 1**: Installs dependencies (5–10 minutes).
     - **Cell 2**: Sets up ngrok and imports.
     - **Cell 3**: Writes `app.py` and `index.html`.
     - **Cell 4**: Starts ngrok and runs the Flask app.
   - Monitor the output of each cell for errors.

3. When Cell 4 runs, it will print a public URL (e.g., `Flask app is running at: https://<random>.ngrok-free.app`).

### 6. Access the Application
1. Open the `ngrok` URL printed in the output (e.g., `https://<random>.ngrok-free.app`) in a web browser.
2. The PDF Summarizer interface will load, showing a form to upload a PDF and specify summary length.
3. Upload a PDF file (<10MB, text-based or scanned) and enter a summary length (50–500 words).
4. Click **Summarize**:
   - If successful, the summary appears in a blue box.
   - If an error occurs (e.g., invalid file, no text extracted), an error message appears in a red box.

### 7. Test with a PDF
- **Via Web Interface**:
  - Use the form to upload a PDF and test summarization.
  - Start with a small text-based PDF (<1MB) to verify functionality.
  - Test a scanned PDF to check OCR (if `pytesseract` is working).

- **Via Colab (Alternative)**:
  - Upload a PDF to Colab’s `/content` directory:
    ```python
    from google.colab import files
    uploaded = files.upload()  # Upload a PDF file
    ```
  - Test the `/summarize` endpoint programmatically:
    ```python
    import requests
    public_url = "https://<your-ngrok-url>"  # Replace with the URL from Cell 4
    with open('/content/sample.pdf', 'rb') as f:
        files = {'pdf': f}
        data = {'max_words': '200'}
        response = requests.post(f'{public_url}/summarize', files=files, data=data)
        print(response.json())
    ```

### 8. Verify GPU Usage
- The app logs whether it’s using the GPU or CPU (check Colab output for `Using GPU` or `Using CPU`).
- Confirm GPU availability:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should print True
  !nvidia-smi
  ```

## Troubleshooting

### ngrok Connection Issues
- **Invalid authtoken**: Ensure the ngrok authtoken in the code is correct.
- **Too many connections**: Free ngrok accounts have limits. Disconnect and reconnect:
  ```python
  ngrok.kill()
  public_url = ngrok.connect(5000, bind_tls=True).public_url
  print(f"New URL: {public_url}")
  ```
- Upgrade to a paid ngrok plan if limits persist.

### Dependency Installation Fails
- Rerun the installation cell. If it fails:
  ```bash
  !rm -rf /content/*
  !pip install flask pyngrok pdfplumber transformers torch pytesseract pillow gunicorn
  ```
- Check disk space:
  ```bash
  !df -h
  ```

### GPU Not Detected
- Verify GPU runtime is enabled (Edit > Notebook settings > Hardware accelerator > GPU).
- Check GPU status:
  ```python
  import torch
  print(torch.cuda.is_available())
  !nvidia-smi
  ```

### Tesseract/OCR Issues
- Confirm Tesseract is installed:
  ```bash
  !tesseract --version
  ```
- Test OCR:
  ```python
  import pytesseract
  print(pytesseract.get_tesseract_version())
  ```

### Model Loading Fails
- Clear the Hugging Face cache:
  ```bash
  !rm -rf ~/.cache/huggingface
  ```
- Ensure Colab has internet access to download `sshleifer/distilbart-cnn-6-6`.
- If memory issues persist, force CPU usage by editing `app.py` (in Cell 3):
  ```python
  device = -1
  dtype = torch.float32
  ```

### Gunicorn Crashes
- Check Gunicorn logs in the Colab output.
- Restart the app:
  ```bash
  !pkill gunicorn
  !gunicorn --bind 0.0.0.0:5000 app:app
  ```
- Alternatively, use Flask’s development server:
  ```python
  !python /content/app.py
  ```

### Slow Summarization
- Large PDFs may be slow due to synchronous processing. Test with small PDFs (<1MB).
- Ensure GPU is enabled to speed up model inference.

## Notes
- **Colab Limitations**:
  - The app runs synchronously without Celery/Redis due to Colab’s ephemeral environment.
  - Sessions disconnect after ~12 hours (free tier). Use Colab Pro for longer sessions or more GPU resources.
  - Free GPU usage is limited; monitor quotas.
- **Production**:
  - Colab is for testing, not production. For production, deploy to platforms like Render or RunPod with Celery and Redis.
- **Google Drive Integration**:
  - To process PDFs from Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    # Access PDFs in /content/drive/MyDrive
    ```

## Support
For issues or additional features (e.g., async processing, production deployment), share error logs or requirements with your support contact.