import re

def parse_user_description(text):
    info = {
        "task_type": "text_classification",
        "total_layers": 3,
        "output_type": "binary",
        "num_classes": 2,
        "dropout_rate": 0.3,
        "rnn_type": "LSTM",
        "activation": "relu",
        "pooling_type": "max",
        "seq_len": 100,
        "vocab_size": 10000,
        "embed_dim": 128,
        "img_height": 64,
        "img_width": 64,
        "channels": 3,
        "num_features": 10,
        "signal_length": 128
    }

    text_lower = text.lower()
    tasks = {
        "text": "text_classification",
        "image": "image_classification",
        "regression": "regression",
        "time series": "time_series",
        "audio": "audio_classification"
    }

    for key in tasks:
        if key in text_lower:
            info["task_type"] = tasks[key]
            break

    match = re.search(r'(\d+)\s*layers?', text_lower)
    if match:
        info["total_layers"] = int(match.group(1))

    if "multiclass" in text_lower or "multi-class" in text_lower:
        info["output_type"] = "multiclass"

    num_classes_match = re.search(r'(\d+)\s*(output classes|classes|categories)', text_lower)
    if num_classes_match:
        info["output_type"] = "multiclass"
        info["num_classes"] = int(num_classes_match.group(1))
    elif "binary" in text_lower:
        info["output_type"] = "binary"
        info["num_classes"] = 2
    elif "regression" in text_lower:
        info["output_type"] = "regression"

    match = re.search(r'dropout\s*([\d\.]+)', text_lower)
    if match:
        info["dropout_rate"] = float(match.group(1))

    if "gru" in text_lower:
        info["rnn_type"] = "GRU"
    elif "lstm" in text_lower:
        info["rnn_type"] = "LSTM"

    for act in ["relu", "leakyrelu", "tanh"]:
        if act in text_lower:
            info["activation"] = act
            break

    for pool in ["max", "average"]:
        if pool in text_lower:
            info["pooling_type"] = pool
            break

    match_patterns = {
        "seq_len": r'sequence length\s*(\d+)',
        "vocab_size": r'vocab(ulary)? size\s*(\d+)',
        "embed_dim": r'embedding dimension\s*(\d+)',
        "img_height": r'image height\s*(\d+)',
        "img_width": r'image width\s*(\d+)',
        "channels": r'channels\s*(\d+)',
        "num_features": r'number of features\s*(\d+)',
        "signal_length": r'signal length\s*(\d+)'
    }

    for key, pattern in match_patterns.items():
        match = re.search(pattern, text_lower)
        if match:
            info[key] = int(match.group(2) if key == "vocab_size" else match.group(1))

    return info
