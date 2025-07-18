from description_parser import parse_user_description
from model_builder import (
    build_text_classification_model,
    build_image_classification_model,
    build_regression_model,
    build_time_series_model,
    build_audio_classification_model
)
from utils import (
    save_and_load_model,
    get_model_summary_str,
    generate_model_description,
    translate_text,
    SUPPORTED_LANGUAGES
)

def main():
    print("Describe your model architecture in one sentence, e.g.:")
    print("Text classification with 5 layers, GRU, dropout 0.2, 3 output classes.")

    user_input = input("\nEnter model description: ")

    print("\nSupported languages:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"{code}: {name}")

    while True:
        lang = input("\nEnter 2-letter language code (e.g., 'hi' for Hindi) or press Enter for English: ").strip().lower()
        if not lang:
            lang = "en"
            break
        if lang in SUPPORTED_LANGUAGES:
            break
        print(f"Error: '{lang}' is not supported. Please choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}")

    params = parse_user_description(user_input)

    print("\nParsed parameters:")
    for k, v in params.items():
        print(f" {k}: {v}")

    task = params["task_type"]
    model_builders = {
        "text_classification": build_text_classification_model,
        "image_classification": build_image_classification_model,
        "regression": build_regression_model,
        "time_series": build_time_series_model,
        "audio_classification": build_audio_classification_model
    }

    model = model_builders.get(task, build_text_classification_model)(params)
    loaded_model = save_and_load_model(model)

    print("\n✅ Generated model architecture:\n")
    summary_text = get_model_summary_str(loaded_model)
    print(summary_text)

    description = generate_model_description(params)
    print("\nModel Description (English):")
    print(description)

    if lang != "en":
        print(f"\nTranslating to {SUPPORTED_LANGUAGES[lang]}...")
        try:
            print(f"\nModel Summary ({SUPPORTED_LANGUAGES[lang]}):")
            print(translate_text(summary_text, dest_lang=lang))
            print(f"\nModel Description ({SUPPORTED_LANGUAGES[lang]}):")
            print(translate_text(description, dest_lang=lang))
        except Exception as e:
            print(f"Translation failed: {e}\nShowing English version instead.")

if __name__ == "__main__":
    main()
