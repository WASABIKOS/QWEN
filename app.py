from flask import Flask, request, jsonify
import requests
import os
import time
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

Z_AI_API_URL = os.getenv('Z_AI_API_URL', 'http://localhost:8000')
Z_AI_API_KEY = os.getenv('Z_AI_API_KEY', '')


@app.route('/', methods=['GET'])
def root():
    return jsonify({'service': 'qwen-zai-bridge', 'status': 'running'}), 200


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/api/z-ai', methods=['POST'])
def z_ai_proxy():
    """
    Proxy entre server.ts (AI Studio app) y la API real de Z.ai.

    Contrato de entrada (lo que server.ts envia, linea 461-465):
        { "message": str, "context": str|None, "model": str }

    Contrato de salida esperado por server.ts (linea 493-501):
        { "response": str, "metadata": { "tokens_used": int } }
    """
    try:
        data = request.json or {}
        message = data.get('message')
        context = data.get('context')
        model = data.get('model', 'z-ai-latest')

        if not message or not isinstance(message, str) or not message.strip():
            return jsonify({
                'status': 'error',
                'error': 'El mensaje no puede estar vacio',
                'code': 'EMPTY_MESSAGE'
            }), 400

        if not Z_AI_API_KEY:
            return jsonify({
                'status': 'error',
                'error': 'Z_AI_API_KEY no configurada en el entorno de Render.',
                'code': 'UNAUTHORIZED'
            }), 401

        headers = {
            'Authorization': f'Bearer {Z_AI_API_KEY}',
            'Content-Type': 'application/json'
        }

        # Ajustar el body/endpoint exacto segun la doc real de la API de Z.ai.
        # Este shape (messages: [{role, content}]) es el formato chat estandar
        # OpenAI-compatible que Z.ai expone; confirmar contra su documentacion.
        upstream_payload = {
            'model': model,
            'messages': [
                *([{'role': 'system', 'content': context}] if context else []),
                {'role': 'user', 'content': message}
            ]
        }

        start = time.time()
        upstream = requests.post(
            f'{Z_AI_API_URL}/api/chat',
            json=upstream_payload,
            headers=headers,
            timeout=25  # debajo del timeout de 30s que aplica server.ts
        )
        latency_ms = int((time.time() - start) * 1000)

        if not upstream.ok:
            if upstream.status_code in (401, 403):
                return jsonify({
                    'status': 'error',
                    'error': 'Error de autenticacion con Z.ai. Verifica Z_AI_API_KEY.',
                    'code': 'AUTHENTICATION_FAILED'
                }), upstream.status_code

            return jsonify({
                'status': 'error',
                'error': f'Z.ai API devolvio {upstream.status_code}: {upstream.text}',
                'code': 'UPSTREAM_ERROR'
            }), 502

        upstream_data = upstream.json()

        # Normalizar la respuesta de Z.ai al contrato que espera server.ts.
        # Ajustar estos paths segun el shape real de respuesta de Z.ai.
        response_text = (
            upstream_data.get('response')
            or upstream_data.get('choices', [{}])[0].get('message', {}).get('content')
            or ''
        )
        tokens_used = (
            upstream_data.get('tokens_used')
            or upstream_data.get('usage', {}).get('total_tokens', 0)
        )

        return jsonify({
            'status': 'success',
            'response': response_text,
            'metadata': {
                'tokens_used': tokens_used,
                'latency_ms': latency_ms
            }
        }), 200

    except requests.exceptions.Timeout:
        return jsonify({
            'status': 'error',
            'error': 'Timeout al conectar con la API de Z.ai (25s).',
            'code': 'TIMEOUT_ERROR'
        }), 408

    except requests.exceptions.ConnectionError as e:
        return jsonify({
            'status': 'error',
            'error': f'No se pudo conectar a Z_AI_API_URL ({Z_AI_API_URL}): {str(e)}',
            'code': 'CONNECTION_FAILED'
        }), 502

    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'code': 'INTERNAL_ERROR'
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
