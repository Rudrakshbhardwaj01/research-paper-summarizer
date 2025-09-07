# Research Paper Summarizer

A Streamlit app to generate concise summaries of research papers, with selectable explanation style and length, powered by Hugging Face LLMs via the Inference API.

---

## Features

- Select from popular research papers
- Choose explanation style:
  - Beginner-Friendly
  - Code-Heavy
  - Mathematically Intuitive
  - Advanced
- Choose summary length: Short, Medium, Long
- Uses Hugging Face language models
- Streamlit interface

---

## Folder Structure

```
research-paper-summarizer/
├── researchPaperSummarizer/
│   ├── __init__.py
│   ├── researchSummarizer.py
│   └── template.json
├── .env
├── promptGenerator.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Prerequisites

- Python 3.10+
- [Hugging Face account](https://huggingface.co)
- Hugging Face Inference API token

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Environment Setup

Create a `.env` file in the root directory with your Hugging Face API token:

```
HF_TOKEN=your_huggingface_api_token_here
```

---

## Clone the Repository

```bash
git clone https://github.com/Rudrakshbhardwaj01/research-paper-summarizer.git
cd research-paper-summarizer
```

---

## Run the App

```bash
cd researchPaperSummarizer
streamlit run researchSummarizer.py
```

Visit [http://localhost:8501](http://localhost:8501)

---

## Usage

- Select a paper
- Choose explanation style and length
- Click "Summarize"

---

## Notes

- Ensure sufficient Hugging Face Inference API credits
- For free-tier use, consider switching to a smaller model:

  ```python
  llm = HuggingFaceEndpoint(
      repo_id="tiiuae/falcon-7b-instruct",
      task="text-generation",
      huggingfacehub_api_token=os.getenv("HF_TOKEN")
  )
  ```

---

## Dependencies

- streamlit
- python-dotenv
- langchain-core
- langchain-huggingface
- huggingface-hub

Install with:

```bash
pip install streamlit python-dotenv langchain-core langchain-huggingface huggingface-hub
```

---

## License

MIT License. See LICENSE.
