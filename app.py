from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

GLM_API_URL = os.getenv('GLM_API_URL', 'http://localhost:8000')
GLM_API_KEY = os.getenv('GLM_API_KEY', '')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/glm', methods=['POST'])
def glm_proxy():
    """Proxy requests from AI Studio to GLM 5.2 API"""
    try:
        data = request.json
        headers = {
            'Authorization': f'Bearer {GLM_API_KEY}',
            'Content-Type': 'application/json'
        }
        response = requests.post(f'{GLM_API_URL}/api/chat', json=data, headers=headers, timeout=30)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot-glm52', methods=['POST'])
def bot_glm52():
    """Bot intermediario para GLM 5.2"""
    try:
        payload = request.json
        result = {
            'status': 'success',
            'message': 'Bot GLM 5.2 processing...',
            'data': payload
        }
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
