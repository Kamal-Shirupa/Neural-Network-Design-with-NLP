# 🧠 Neural Network Design With NLP

This project allows users to design and generate neural network architectures using natural language descriptions. It supports multiple machine learning tasks—such as text classification, image classification, regression, and more—and provides outputs and summaries in various languages.

## 🚀 Features

- **Natural language model design:** Describe model requirements in plain English, and automatically generate appropriate architectures.
- **Support for multiple ML modalities:** Text, image, audio, regression, and time series tasks.
- **Multilingual summaries:** Outputs and model descriptions can be generated in several languages.
- **Modular architecture:** Clean codebase with well-organized modules for parsing, model building, and utilities.
- **Reproducible examples:** Includes sample inputs, outputs, and a demo Jupyter notebook.

## 📂 Project Structure

```
neural-network-nlp/
├── src/           # Core source code
├── config/        # Default values and language configs
├── requirements/  # Python dependencies
├── samples/       # Sample inputs & outputs
├── notebooks/     # Demo and example notebooks
├── .gitignore
├── LICENSE
├── README.md
```

## 🛠️ Requirements

Install all dependencies:

```
pip install -r requirements/requirements.txt
```

Key packages:
- **tensorflow:** Deep learning framework
- **autokeras:** Automated model building
- **googletrans:** For multilingual translation

## 📄 Sample Input

```
10 layers, GRU, dropout 0.3, 5 output classes, channels 4
```

## 📑 Sample Output

<img width="700" height="594" alt="image" src="https://github.com/user-attachments/assets/1d519d27-64c5-40bc-9728-6077d3cca719" />


## 🌍 Supported Languages

- English (`en`)
- French (`fr`)
- Spanish (`es`)
- German (`de`)
- Hindi (`hi`)
- Telugu (`te`)
- Japanese (`ja`)
- Chinese Simplified (`zh-cn`)
- Russian (`ru`)

## 🏗️ Configuration

- **Default model parameters** and **supported language codes** can be edited under the [`config/`](config/) folder.

## 📖 License

Distributed under the MIT License.

## 👤 Author

Kamal-Shirupa

**For a detailed walkthrough, see the sample files and demo notebook. If you have questions or want to contribute, please create an issue or pull request.**
