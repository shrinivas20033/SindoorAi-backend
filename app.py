from together import Together
# Proper imports
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
IMGBB_API_KEY = os.getenv('IMGBB_API_KEY')
TOGETHERAI_API_KEY = os.getenv('TOGETHERAI_API_KEY')

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# /chat endpoint for chat and chat-with-image, using OpenAI GPT-4o mini and imgbb for image upload
@app.route('/chat', methods=['POST','GET'])
def chat():
	try:
		data = request.get_json(force=True)
		user_query = data.get('query')
		image_data = data.get('image')  # Expecting base64 string if present

		# Accept either base64 image, direct imgbb URL, or 'image_url' key
		image_url = None
		# Support both 'image' (base64 or url) and 'image_url' (url) keys
		if data.get('image_url'):
			image_url = data['image_url']
		elif image_data:
			if isinstance(image_data, str) and image_data.strip().startswith('http'):
				image_url = image_data.strip()
			else:
				imgbb_url = 'https://api.imgbb.com/1/upload'
				payload = {
					'key': IMGBB_API_KEY,
					'image': image_data
				}
				imgbb_response = requests.post(imgbb_url, data=payload)
				if imgbb_response.status_code == 200:
					image_url = imgbb_response.json()['data']['display_url']
				else:
					return jsonify({'error': 'Image upload failed', 'details': imgbb_response.text}), 400

		# Prepare messages for OpenAI GPT-4o mini
		messages = []
		if user_query:
			messages.append({'role': 'user', 'content': user_query})
		if image_url:
			# Pass image as an image block, not as text
			messages.append({'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image_url}}]})

		# Call OpenAI GPT-4o mini model (use correct model name for mini)
		openai_url = 'https://api.openai.com/v1/chat/completions'
		headers = {
			'Authorization': f'Bearer {OPENAI_API_KEY}',
			'Content-Type': 'application/json'
		}
		payload = {
			'model': 'gpt-4o-mini',
			'messages': messages
		}
		openai_response = requests.post(openai_url, headers=headers, json=payload)
		if openai_response.status_code == 200:
			result = openai_response.json()
			answer = result['choices'][0]['message']['content']
			return jsonify({'response': answer, 'image_url': image_url})
		else:
			return jsonify({'error': 'OpenAI API failed', 'details': openai_response.text}), 400
	except Exception as e:
		return jsonify({'error': 'Server error', 'details': str(e)}), 500



from werkzeug.utils import secure_filename
from ml_integration.resume import process_resume
# /resume-analyze endpoint for resume file upload and ML prediction using local embedding model
@app.route('/resume-analyze', methods=['POST'])
def resume_analyze():
	try:
		if 'file' not in request.files:
			return jsonify({'error': 'No file part in the request'}), 400
		file = request.files['file']
		if file.filename == '':
			return jsonify({'error': 'No selected file'}), 400
		filename = secure_filename(file.filename)
		temp_path = os.path.join('ml_models', filename)
		file.save(temp_path)

		# Process the resume to get embedding, label, and text
		result = process_resume(temp_path)

		# Get prediction probability (percentage) for the top role
		import numpy as np
		from ml_integration import resume as resume_module
		label_encoder = resume_module.joblib.load(resume_module.LABEL_ENCODER_PATH)
		xgb_model = resume_module.joblib.load(resume_module.XGB_MODEL_PATH)
		embedding = result['embedding']
		proba = xgb_model.predict_proba([embedding])[0]
		top_idx = np.argmax(proba)
		top_label = label_encoder.inverse_transform([top_idx])[0]
		top_percent = round(float(proba[top_idx]) * 100, 2)

		label = result['label']
		text = result['text']
		prompt = f"""
You are an expert career coach and hiring manager with 20 years of experience. Analyze the following resume text and the predicted job role. Provide:
1. A brief explanation of why the model predicted this role.
2. The strengths of the resume for this role.
3. The weaknesses of the resume for this role.
4. Advice for improving the resume for the top 2 most likely roles (the predicted role and one other likely role, e.g., Data Scientist and Backend Engineer). Suggest specific improvements for each role.

Resume text:
{text}

Predicted top role: {label} ({top_percent}%)
"""

		client = Together(api_key=os.getenv('TOGETHERAI_API_KEY'))
		llm_response = client.chat.completions.create(
			model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
			messages=[
				{"role": "user", "content": prompt}
			]
		)
		llm_content = llm_response.choices[0].message.content if hasattr(llm_response, 'choices') and llm_response.choices else None

		return jsonify({'llm_analysis': llm_content, 'role': label, 'percentage': top_percent})
	except Exception as e:
		return jsonify({'error': 'Server error', 'details': str(e)}), 500


if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port, debug=False)
