# LocalRAG: Offline RAG System with Mistral

A lightweight, fully local Retrieval-Augmented Generation (RAG) system that runs entirely on your machine. Built with Mistral 7B, FAISS, and Sentence Transformers for powerful offline document Q&A.

## Features

- 100% offline - no internet required after setup
- Process text from files or web URLs
- Uses Mistral 7B for powerful local inference
- FAISS for fast similarity search
- Sentence Transformers for semantic understanding
- Clean command-line interface

## 📋 Prerequisites

1. **Python 3.8+**
2. **Visual Studio Build Tools 2022** (for Windows)
   - Download from: [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Install with "Desktop development with C++" workload
   - Required for compiling `llama-cpp-python`

3. **At least 8GB RAM** (16GB recommended)
4. **4GB+ free disk space** for the model

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/laibashakil/LocalRAG.git
cd LocalRAG
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the Mistral model**
- Download [mistral-7b-instruct-v0.1.Q4_K_M.gguf](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf)
- Place it in the `models` directory
- Create the directory if it doesn't exist: `mkdir models`

## 💻 Usage

### 1. Using a Text File
```bash
python rag_main.py --file text.txt --model_path models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 2. Using a Web URL
```bash
python rag_main.py --url "https://example.com/article" --model_path models/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

### 3. Ask Questions
Once the system is running, you can ask questions about the content. The system will:
1. Find relevant context from the text
2. Generate an answer using Mistral
3. Display the response

Type 'exit' to quit the program.

## Project Structure

```
.
├── models/                  # Directory for Mistral model
│   └── mistral-7b-instruct-v0.1.Q4_K_M.gguf
├── utils/
│   ├── embedder.py         # FAISS + Sentence Transformers
│   ├── file_loader.py      # Text file processing
│   ├── web_scraper.py      # URL content extraction
│   └── mistral_interface.py # Mistral model interface
├── rag_main.py             # Main script
├── requirements.txt        # Dependencies
└── README.md
```

## 🔧 Dependencies

- `llama-cpp-python`: For running Mistral locally
- `sentence-transformers`: For text embeddings
- `faiss-cpu`: For similarity search
- `beautifulsoup4`: For web scraping
- `requests`: For HTTP requests

## ⚙️ Configuration

The system is optimized for speed and efficiency:
- Context window: 1024 tokens
- Batch size: 512
- Temperature: 0.7
- Top-k: 40
- Top-p: 0.9

## Troubleshooting

1. **Model loading issues**
   - Ensure you have enough RAM (8GB+)
   - Check if the model file is in the correct location
   - Verify the model file isn't corrupted

2. **Build tools issues**
   - Make sure Visual Studio Build Tools are properly installed
   - Try running in the Developer Command Prompt for VS 2022

3. **Performance issues**
   - Reduce the context window size
   - Use a smaller model variant
   - Close other memory-intensive applications

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Mistral AI](https://mistral.ai/) for the base model
- [FAISS](https://github.com/facebookresearch/faiss) for similarity search
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for model inference