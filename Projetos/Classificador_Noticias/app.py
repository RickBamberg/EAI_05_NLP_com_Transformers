"""
Classificador de Notícias com BERT Fine-tunado
Flask Web Application

Autor: Carlos Henrique Bamberg Marques
Módulo: EAI_05 - NLP com Transformers
Data: Janeiro 2026
"""

from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from logging.handlers import RotatingFileHandler
import os

# ========== CONFIGURAÇÃO DO APP ==========
app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# ========== LOGGING ==========
if not os.path.exists('logs'):
    os.makedirs('logs')

handler = RotatingFileHandler(
    'logs/app.log',
    maxBytes=10000000,  # 10MB
    backupCount=5
)

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
app.logger.addHandler(handler)
app.logger.setLevel(logging.INFO)

# ========== CARREGAR MODELO ==========
MODEL_PATH = 'models/bert_news_classifier'

# Verificar se modelo existe
if not os.path.exists(MODEL_PATH):
    app.logger.error(f"Modelo não encontrado em {MODEL_PATH}")
    app.logger.info("Por favor, treine o modelo primeiro executando o notebook.")
    # Usar modelo base como fallback (sem fine-tuning)
    MODEL_PATH = 'distilbert-base-uncased'
    app.logger.warning("Usando modelo base sem fine-tuning como fallback")

app.logger.info("Carregando tokenizer e modelo...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    modelo = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    modelo.eval()
    
    # GPU se disponível
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(device)
    
    app.logger.info(f"Modelo carregado com sucesso em {device}")
except Exception as e:
    app.logger.error(f"Erro ao carregar modelo: {e}")
    raise

# ========== MAPEAMENTO DE CLASSES ==========
CLASSES = {
    0: 'World',
    1: 'Sports', 
    2: 'Business',
    3: 'Sci/Tech'
}

CLASS_DESCRIPTIONS = {
    'World': 'Notícias internacionais, política, diplomacia',
    'Sports': 'Esportes, competições, atletas',
    'Business': 'Negócios, economia, mercados',
    'Sci/Tech': 'Ciência, tecnologia, inovação'
}

CLASS_EMOJIS = {
    'World': '🌍',
    'Sports': '⚽',
    'Business': '💼',
    'Sci/Tech': '🔬'
}

# ========== FUNÇÃO DE CLASSIFICAÇÃO ==========
def classificar_noticia(texto, top_k=3):
    """
    Classifica uma notícia usando o modelo BERT fine-tunado
    
    Args:
        texto (str): Texto da notícia
        top_k (int): Número de top predições a retornar
    
    Returns:
        dict: {
            'classe_principal': str,
            'confianca_principal': float,
            'top_predicoes': list[dict]
        }
    """
    try:
        # Tokenizar
        inputs = tokenizer(
            texto,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        # Predição
        with torch.no_grad():
            outputs = modelo(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu()
        
        # Top-k predições
        top_probs, top_indices = torch.topk(probs, min(top_k, len(CLASSES)))
        
        top_predicoes = []
        for i in range(len(top_probs)):
            classe = CLASSES[top_indices[i].item()]
            top_predicoes.append({
                'posicao': i + 1,
                'classe': classe,
                'emoji': CLASS_EMOJIS[classe],
                'descricao': CLASS_DESCRIPTIONS[classe],
                'confianca': top_probs[i].item(),
                'confianca_pct': f"{top_probs[i].item() * 100:.2f}%"
            })
        
        # Classe principal
        classe_principal = top_predicoes[0]['classe']
        confianca_principal = top_predicoes[0]['confianca']
        
        return {
            'classe_principal': classe_principal,
            'confianca_principal': confianca_principal,
            'top_predicoes': top_predicoes
        }
        
    except Exception as e:
        app.logger.error(f"Erro na classificação: {e}")
        raise

# ========== ROTAS ==========

@app.route('/')
def home():
    """Página inicial"""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """Rota de classificação"""
    try:
        # Receber texto
        texto = request.form.get('news_text', '').strip()
        
        # Validações
        if not texto:
            app.logger.warning("Tentativa de classificação sem texto")
            return render_template('index.html', 
                                 error="❌ Por favor, insira uma notícia para classificar.")
        
        if len(texto) < 20:
            app.logger.warning(f"Texto muito curto: {len(texto)} caracteres")
            return render_template('index.html',
                                 error="❌ Texto muito curto. Por favor, insira pelo menos 20 caracteres.",
                                 texto=texto)
        
        if len(texto) > 5000:
            app.logger.warning(f"Texto muito longo: {len(texto)} caracteres")
            return render_template('index.html',
                                 error="❌ Texto muito longo. Máximo: 5000 caracteres.",
                                 texto=texto[:5000])
        
        # Log
        app.logger.info(f"Classificando notícia: {texto[:100]}...")
        
        # Classificar
        resultado = classificar_noticia(texto, top_k=4)
        
        # Log resultado
        app.logger.info(f"Resultado: {resultado['classe_principal']} ({resultado['confianca_principal']:.2%})")
        
        return render_template('resultado.html',
                             news_text=texto,
                             resultado=resultado)
    
    except Exception as e:
        app.logger.error(f"Erro na rota /classify: {e}", exc_info=True)
        return render_template('index.html',
                             error=f"❌ Erro ao processar: {str(e)}",
                             texto=texto if 'texto' in locals() else '')

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    API REST para classificação
    
    Request:
        POST /api/classify
        Content-Type: application/json
        Body: {"text": "..."}
    
    Response:
        {
            "success": true,
            "data": {
                "class": "Business",
                "confidence": 0.943,
                "top_predictions": [...]
            }
        }
    """
    try:
        # Receber JSON
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo "text" é obrigatório'
            }), 400
        
        texto = data['text'].strip()
        
        # Validações
        if not texto:
            return jsonify({
                'success': False,
                'error': 'Texto vazio'
            }), 400
        
        if len(texto) < 20:
            return jsonify({
                'success': False,
                'error': 'Texto muito curto (mínimo 20 caracteres)'
            }), 400
        
        # Classificar
        resultado = classificar_noticia(texto, top_k=4)
        
        app.logger.info(f"API: {resultado['classe_principal']} ({resultado['confianca_principal']:.2%})")
        
        return jsonify({
            'success': True,
            'data': {
                'class': resultado['classe_principal'],
                'confidence': resultado['confianca_principal'],
                'top_predictions': resultado['top_predicoes']
            }
        })
    
    except Exception as e:
        app.logger.error(f"Erro na API: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'model': 'loaded',
        'device': str(device),
        'classes': list(CLASSES.values())
    })

@app.route('/examples')
def examples():
    """Exemplos de notícias para teste"""
    exemplos = {
        'World': "UN Security Council holds emergency meeting to discuss escalating tensions in the Middle East. International leaders call for immediate diplomatic resolution.",
        'Sports': "Lakers defeat Celtics 108-102 in overtime thriller at TD Garden. LeBron James leads with 35 points and 12 assists in crucial playoff game.",
        'Business': "Stock market reaches all-time high as tech sector soars. Apple and Microsoft lead gains with record quarterly earnings beating analyst expectations.",
        'Sci/Tech': "Scientists discover new exoplanet in habitable zone. Advanced telescopes reveal Earth-sized world orbiting nearby star with potential for liquid water."
    }
    
    return jsonify({
        'success': True,
        'examples': exemplos
    })

# ========== ERROR HANDLERS ==========

@app.errorhandler(404)
def not_found(error):
    """Página não encontrada"""
    return render_template('index.html', 
                         error="❌ Página não encontrada"), 404

@app.errorhandler(500)
def internal_error(error):
    """Erro interno do servidor"""
    app.logger.error(f"Erro 500: {error}")
    return render_template('index.html',
                         error="❌ Erro interno do servidor. Tente novamente."), 500

@app.errorhandler(413)
def request_entity_too_large(error):
    """Payload muito grande"""
    return render_template('index.html',
                         error="❌ Texto muito grande. Máximo: 16MB"), 413

# ========== MAIN ==========

if __name__ == '__main__':
    app.logger.info("=" * 50)
    app.logger.info("Iniciando Classificador de Notícias")
    app.logger.info(f"Modelo: {MODEL_PATH}")
    app.logger.info(f"Device: {device}")
    app.logger.info(f"Classes: {list(CLASSES.values())}")
    app.logger.info("=" * 50)
    
    # Modo debug (apenas desenvolvimento)
    # Em produção, usar Gunicorn:
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )
