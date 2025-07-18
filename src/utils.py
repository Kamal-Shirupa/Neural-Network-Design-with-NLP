import os
import io
from contextlib import redirect_stdout
from tensorflow.keras.models import load_model
from googletrans import Translator

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'hi': 'Hindi',
    'te': 'Telugu',
    'ja': 'Japanese',
    'zh-cn': 'Chinese (Simplified)',
    'ru': 'Russian'
}

def save_and_load_model(model, save_path="model.keras"):
    if os.path.exists(save_path):
        os.remove(save_path)
    model.save(save_path)
    print(f"\nModel saved to '{save_path}'.")
    return load_model(save_path)

def get_model_summary_str(model):
    stream = io.StringIO()
    with redirect_stdout(stream):
        model.summary()
    return stream.getvalue()

def generate_model_description(params):
    description = f"This is a {params['task_type'].replace('_', ' ')} model with:\n"
    description += f"- {params['total_layers']} total layers\n"
    description += f"- {params['output_type']} output type\n"
    if params['output_type'] in ['binary', 'multiclass']:
        description += f"- {params['num_classes']} output classes\n"
    description += f"- Dropout rate: {params['dropout_rate']}\n"
    description += f"- Activation function: {params['activation']}\n"

    if params["task_type"] == "text_classification":
        description += f"- Using {params['rnn_type']} layers\n"
        description += f"- Sequence length: {params['seq_len']}\n"
        description += f"- Vocabulary size: {params['vocab_size']}\n"
        description += f"- Embedding dimension: {params['embed_dim']}\n"
    elif params["task_type"] == "image_classification":
        description += f"- Image dimensions: {params['img_height']}x{params['img_width']}x{params['channels']}\n"
    elif params["task_type"] == "time_series":
        description += f"- Sequence length: {params['seq_len']}\n"
        description += f"- Number of features: {params['num_features']}\n"
    elif params["task_type"] == "audio_classification":
        description += f"- Signal length: {params['signal_length']}\n"
        description += f"- Channels: {params['channels']}\n"
    elif params["task_type"] == "regression":
        description += f"- Number of features: {params['num_features']}\n"

    return description.strip()

def translate_text(text, dest_lang="en"):
    translator = Translator()
    try:
        return translator.translate(text, dest=dest_lang).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text
