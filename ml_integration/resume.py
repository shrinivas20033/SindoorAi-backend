from sentence_transformers import SentenceTransformer
import re
# Resume processing pipeline: extract text, encode, and predict
import os

import joblib
import pdfplumber
import docx
import requests

ML_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ml_models'))
LABEL_ENCODER_PATH = os.path.join(ML_MODELS_DIR, 'label_encoder.pkl')
XGB_MODEL_PATH = os.path.join(ML_MODELS_DIR, 'xgboost_resume_model.pkl')

def extract_text_from_file(file_path):
	"""
	Extract text from PDF, DOC, or DOCX file using pdfplumber or python-docx.
	"""
	ext = file_path.split('.')[-1].lower()
	try:
		if ext == 'pdf':
			with pdfplumber.open(file_path) as pdf:
				text = "\n".join(page.extract_text() or '' for page in pdf.pages)
			return text
		elif ext in ('docx', 'doc'):
			doc = docx.Document(file_path)
			text = "\n".join([para.text for para in doc.paragraphs])
			return text
		else:
			raise RuntimeError('Unsupported file type')
	except Exception as e:
		raise RuntimeError(f"Failed to extract text: {e}")



def encode_text_with_online_nlp(text):
	"""
	Encode text using OpenAI online embedding API (text-embedding-3-small).
	Returns embedding vector.
	"""
	import os
	import requests
	import numpy as np
	OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
	if not OPENAI_API_KEY:
		raise RuntimeError('OPENAI_API_KEY not set in environment')

	# Split text into sentences (simple split, can be improved)
	sentences = [s.strip() for s in re.split(r'[\n\.!?]', text) if s.strip()]
	if not sentences:
		raise RuntimeError("No sentences found in text for embedding.")

	# For long docs, embed each chunk and average
	def chunk_sentences(sent_list, max_words=300):
		chunks = []
		current = []
		count = 0
		for s in sent_list:
			words = s.split()
			idx = 0
			while idx < len(words):
				remaining = max_words - count
				take = min(remaining, len(words) - idx)
				if take == 0:
					chunks.append(" ".join(current))
					current = []
					count = 0
					remaining = max_words
					take = min(remaining, len(words) - idx)
				current.extend(words[idx:idx+take])
				count += take
				idx += take
				if count >= max_words:
					chunks.append(" ".join(current))
					current = []
					count = 0
		if current:
			chunks.append(" ".join(current))
		return chunks

	text_chunks = chunk_sentences(sentences)
	embeddings = []
	for chunk in text_chunks:
		# Call OpenAI embedding API for each chunk
		response = requests.post(
			'https://api.openai.com/v1/embeddings',
			headers={
				'Authorization': f'Bearer {OPENAI_API_KEY}',
				'Content-Type': 'application/json'
			},
			json={
				'input': chunk,
				'model': 'text-embedding-3-small'
			}
		)
		if response.status_code == 200:
			emb = response.json()['data'][0]['embedding']
			embeddings.append(emb)
		else:
			raise RuntimeError(f"OpenAI embedding API failed: {response.text}")
	avg_emb = np.mean(embeddings, axis=0)
	return avg_emb.tolist()

def predict_resume_label(embedding):
	"""
	Predict using the loaded ML model and label encoder.
	"""
	label_encoder = joblib.load(LABEL_ENCODER_PATH)
	xgb_model = joblib.load(XGB_MODEL_PATH)
	# XGBoost expects 2D array
	pred = xgb_model.predict([embedding])
	label = label_encoder.inverse_transform(pred)
	return label[0]

def process_resume(file_path):
	"""
	Full pipeline: extract text, encode, and predict label using online embedding model.
	"""
	text = extract_text_from_file(file_path)
	embedding = encode_text_with_online_nlp(text)
	label = predict_resume_label(embedding)
	return {'label': label, 'text': text, 'embedding': embedding}
