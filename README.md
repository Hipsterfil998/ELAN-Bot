# ELAN-Bot 🤖

ELAN-Bot is a specialized chatbot designed to assist users with ELAN, the professional annotation software widely used in linguistics, language documentation, and multimodal research. This virtual assistant provides guidance on how to use ELAN's features and can help modify EAF files through natural language interactions.

Try to chat with [ELAN-Bot](https://huggingface.co/spaces/HipFil98/ELAN_bot) to check its functionalities!
Available in English 🇬🇧, Spanish 🇪🇸, Italian 🇮🇹 and French 🇫🇷

<img src="elan_bot.png" alt="Chat with ELAN-BOT"/>

## About ELAN

ELAN (EUDICO Linguistic Annotator) is a professional tool for the creation of complex annotations on video and audio resources, developed by the Max Planck Institute for Psycholinguistics. It's widely used by researchers and professionals in linguistics, language documentation, sign language research, gesture studies, and multimodal interaction analysis.

## Features

- **Knowledge-Based Assistance**: Get answers to questions about ELAN's functionality and features
- **XML Code Modification**: Provide XML code from EAF files for automated modifications
- **Vector Search**: Powered by Qdrant and Sentence Transformers for intelligent retrieval
- **LLM Integration**: Generates helpful, concise responses using Meta's Llama 3.2 models
- **Interactive Chat Interface**: User-friendly Gradio interface for natural conversations
- **Example Queries**: Built-in example questions to help users get started

## Technical Stack

- **Frontend**: Gradio chat interface
- **Vector Database**: Qdrant for efficient similarity search
- **Embeddings**: Sentence Transformers (Nomic Embed Text v1.5)
- **Language Model**: Meta Llama 3.2 models via Hugging Face Inference API
- **Python Libraries**: Gradio, Qdrant Client, Sentence Transformers, Hugging Face Hub

## Prerequisites

- Python 3.10+
- Hugging Face API key (for Inference API access)
- Sentence Transformers
- Qdrant Client
- Gradio

## Installation

1. Clone the repository:
```bash
https://github.com/Hipsterfil998/ELAN-bot.git
cd ELAN-Bot
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Hugging Face API token:
```bash
export HF_TOKEN="your_huggingface_token"
```

4. Prepare your vector database:
```bash
python ELAN_bot/database/database.py
```

5. Start the ELAN-Bot:
```bash
python app.py
```

6. Access the web interface through your browser at `http://localhost:7860`

## Usage

ELAN-Bot supports two primary modes of interaction:

### 1. ELAN Manual Queries
Ask questions about how to use ELAN and its features:
- "How can I add a new tier in ELAN?"
- "How can I export annotations in txt format?"
- "How can I search within annotations?"

### 2. XML Code Modification
Provide XML code from an EAF file with instructions for modification:
- Paste the XML code and add instructions like "Change the AUTHOR attribute to 'John Smith'"
- Provide the XML code and ask "Extract all tier names from this file"

## Contributing

Contributions are welcome! Feel free to:
- Enhance the knowledge base with more ELAN information
- Improve response quality and accuracy
- Add new features to help ELAN users
- Fix bugs and enhance code quality

Please submit pull requests for any improvements you'd like to contribute.

## License

This project is licensed under the XXX License. See the LICENSE file for details.

## Acknowledgments

- The Max Planck Institute for Psycholinguistics for developing ELAN
- The Hugging Face team for their Inference API
- The developers of Qdrant, Sentence Transformers, and Gradio
- All contributors who help improve this project
