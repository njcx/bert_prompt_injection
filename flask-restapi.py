from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import logging
import time
from functools import wraps


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 全局变量
model = None
tokenizer = None
device = None

# 配置
MODEL_PATH = './prompt_injection_detector'  # 模型路径
MAX_LENGTH = 512
CONFIDENCE_THRESHOLD = 0.7  # 判定阈值
API_KEY = ["H8jR4nPqW6sYtVcE", "T7gFpN3mKx9LcVbQ", "R2vZqW8nJk4PmSxH"]


def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key not in API_KEY:
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated


def load_model():
    """加载模型和分词器"""
    global model, tokenizer, device

    try:
        logger.info("Loading model and tokenizer...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        logger.info("Model loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False


def timing_decorator(f):
    """计算接口响应时间的装饰器"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        elapsed_time = time.time() - start_time

        # 在响应中添加处理时间
        if isinstance(result, tuple):
            response, status_code = result
            if isinstance(response.json, dict):
                response.json['processing_time_ms'] = round(elapsed_time * 1000, 2)

        logger.info(f"{request.path} - {elapsed_time * 1000:.2f}ms")
        return result

    return decorated_function


def predict_injection(text, threshold=CONFIDENCE_THRESHOLD):
    """
    预测文本是否为提示词注入

    Args:
        text: 输入文本
        threshold: 判定阈值

    Returns:
        dict: 预测结果
    """
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    # 文本编码
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 模型推理
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(probs, dim=1).item()
        confidence = probs[0][prediction].item()

    # 构建结果
    is_injection = prediction == 1 and confidence >= threshold

    result = {
        'text': text,
        'is_injection': is_injection,
        'prediction': 'injection' if prediction == 1 else 'normal',
        'confidence': round(confidence, 4),
        'probabilities': {
            'normal': round(probs[0][0].item(), 4),
            'injection': round(probs[0][1].item(), 4)
        }
    }

    return result


@app.route('/predict', methods=['POST'])
@timing_decorator
@require_api_key
def predict():
    """
    预测单个文本

    请求体：
    {
        "text": "忽略之前的指令",
        "threshold": 0.5  # 可选
    }

    响应：
    {
        "success": true,
        "data": {
            "text": "忽略之前的指令",
            "is_injection": true,
            "prediction": "injection",
            "confidence": 0.9876,
            "probabilities": {
                "normal": 0.0124,
                "injection": 0.9876
            }
        },
        "processing_time_ms": 45.23
    }
    """
    try:
        # 检查模型是否加载
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        # 获取请求数据
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: text'
            }), 400

        text = data['text']
        threshold = data.get('threshold', CONFIDENCE_THRESHOLD)

        # 验证输入
        if not isinstance(text, str) or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text must be a non-empty string'
            }), 400

        if len(text) > 10000:
            return jsonify({
                'success': False,
                'error': 'Text too long (max 10000 characters)'
            }), 400

        # 执行预测
        result = predict_injection(text, threshold)

        return jsonify({
            'success': True,
            'data': result
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict/batch', methods=['POST'])
@timing_decorator
@require_api_key
def predict_batch():
    """
    批量预测

    请求体：
    {
        "texts": ["文本1", "文本2", "文本3"],
        "threshold": 0.5  # 可选
    }

    响应：
    {
        "success": true,
        "data": {
            "results": [...],
            "summary": {
                "total": 3,
                "injection_count": 1,
                "normal_count": 2
            }
        },
        "processing_time_ms": 123.45
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 503

        data = request.get_json()

        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: texts'
            }), 400

        texts = data['texts']
        threshold = data.get('threshold', CONFIDENCE_THRESHOLD)

        # 验证输入
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({
                'success': False,
                'error': 'texts must be a non-empty list'
            }), 400

        if len(texts) > 100:
            return jsonify({
                'success': False,
                'error': 'Too many texts (max 100 per request)'
            }), 400

        # 批量预测
        results = []
        injection_count = 0

        for text in texts:
            if isinstance(text, str) and len(text.strip()) > 0:
                result = predict_injection(text, threshold)
                results.append(result)
                if result['is_injection']:
                    injection_count += 1
            else:
                results.append({
                    'text': text,
                    'error': 'Invalid text'
                })

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'summary': {
                    'total': len(results),
                    'injection_count': injection_count,
                    'normal_count': len(results) - injection_count
                }
            }
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """404 错误处理"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """500 错误处理"""
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # 启动时加载模型
    if not load_model():
        logger.error("Failed to load model. Exiting...")
        exit(1)

    # 启动服务
    logger.info("Starting Flask API server...")
    app.run(
        host='0.0.0.0',  # 允许外部访问
        port=5008,
        debug=False,  # 生产环境设为 False
        threaded=True  # 支持多线程
    )