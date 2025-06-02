# Medical Question Answering System using Wikipedia

## Overall Objective

This project aims to develop a system that can answer medical multiple-choice questions from the MedQA dataset. It leverages Wikipedia as its primary knowledge source, dynamically fetching and processing articles to find the best answer. The system is designed to utilize GPU acceleration where possible (especially for neural model inference and FAISS) and employs a vector database (FAISS) for efficient information retrieval.

## Features

*   Processes questions from the MedQA dataset (JSONL format).
*   Uses Wikipedia as a dynamic, external knowledge source.
*   Leverages Sentence-Transformers ("all-MiniLM-L6-v2") for generating high-quality semantic embeddings of questions and text passages.
*   Employs FAISS (Facebook AI Similarity Search) for efficient similarity search in a vector database of text segments (supports both CPU and GPU versions).
*   Utilizes SciSpacy ("en_core_sci_sm") for Named Entity Recognition (NER) tailored for biomedical text.
*   Supports GPU acceleration for PyTorch model inference (CUDA) and FAISS indexing/searching.
*   Modular design, with distinct Python modules for different stages of the QA pipeline.
*   Includes unit tests for core components to ensure reliability.
*   Comprehensive logging throughout the pipeline.

## Directory Structure

```
.
├── main.py                 # Main script to run the QA pipeline
├── src/                    # Contains all core Python modules
│   ├── __init__.py
│   ├── data_loader.py          # Handles MedQA dataset loading
│   ├── question_processor.py   # Processes questions (NER, embedding)
│   ├── wikipedia_selector.py   # Selects and fetches Wikipedia articles
│   ├── evidence_scorer.py      # Segments evidence, builds FAISS index, scores options
│   ├── answer_selector.py      # Selects the final answer based on scores
│   ├── distilbert_classifier.py # Placeholder for question classification model
│   └── distilbert_nli.py       # Placeholder for NLI-based scoring model
├── data/                   # Intended for dataset files (e.g., MedQA). Needs to be created by the user.
│   └── (example: medqa_usmle.jsonl)
├── tests/                  # Contains unit tests
│   ├── __init__.py
│   ├── common.py
│   ├── test_data_loader.py
│   ├── test_question_processor.py
│   ├── test_wikipedia_selector.py (Placeholder - not yet implemented)
│   ├── test_evidence_scorer.py
│   └── test_answer_selector.py
├── requirements.txt        # (Optional - can be generated via pip freeze > requirements.txt)
└── README.md               # This file
```

## Setup Instructions

### Prerequisites
*   Python 3.8+
*   Access to a terminal or command prompt.
*   Git (for cloning the repository).

### 1. Clone Repository
```bash
git clone <repository_url> # Replace <repository_url> with the actual URL
cd <repository_directory_name>
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# For Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
Install PyTorch first, as its installation can be system-specific (CPU-only or specific CUDA versions).
```bash
# Refer to https://pytorch.org/get-started/locally/ for the correct command for your system
pip install torch torchvision torchaudio
```

Then, install other Python packages:
```bash
pip install transformers sentence-transformers
pip install spacy
python -m spacy download en_core_sci_sm
pip install wikipedia beautifulsoup4 requests scikit-learn nltk
```

Finally, install FAISS. Choose one of the following options based on your system:

**Option 1: FAISS with GPU support (Recommended if CUDA is available)**
```bash
# Ensure you have a compatible NVIDIA driver and CUDA toolkit installed.
# Check the FAISS GitHub repository for specific CUDA version compatibility.
# Example for CUDA 11.x (might vary):
# pip install faiss-gpu-cuda11X 
pip install faiss-gpu # Or the specific wheel for your CUDA version
```
*Note: You might need to install FAISS from a specific wheel or build from source depending on your CUDA version. Consult the [official FAISS documentation/GitHub](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).*

**Option 2: FAISS with CPU support**
```bash
pip install faiss-cpu
```

## Dataset (MedQA)

This system is designed to work with the MedQA dataset.
*   You will need to obtain the dataset (e.g., from the official MedQA repository/website or the source where it was provided).
*   The dataset should be in **JSONL (JSON Lines)** format, where each line is a valid JSON object representing a question.
*   Create a `data/` directory in the project root if it doesn't exist.
*   Place your MedQA JSONL file (e.g., `medqa_usmle.jsonl`, `dev.jsonl`, `test.jsonl`) into this `data/` directory.

Each JSON object (line) in the file should conform to the following structure:
```json
{
  "id": "some_unique_id", // Optional, but useful for tracking
  "question": "What is the primary cause of Type 1 Diabetes Mellitus?",
  "options": {
    "A": "Viral infection",
    "B": "Insulin resistance",
    "C": "Autoimmune destruction of beta cells",
    "D": "Obesity"
  },
  "answer_idx": "C", // The key corresponding to the correct option in the "options" dictionary
  "metamap_phrases": ["Type 1 Diabetes Mellitus", "Primary Cause"] // Optional, list of relevant medical concepts
}
```
*   `question`: (string) The medical question.
*   `options`: (dict) A dictionary where keys are option letters (e.g., 'A', 'B', 'C', 'D') and values are the corresponding option texts (string).
*   `answer_idx`: (string) The key from the `options` dictionary that represents the correct answer.
*   `metamap_phrases`: (list of strings, optional) Pre-extracted medical concepts related to the question. If not present, the system will rely solely on NER.

## Running the System

The main script `main.py` is used to run the entire question-answering pipeline.

### Command-line Arguments:

*   `dataset_file_path` (Required, positional): Path to the MedQA JSONL dataset file.
    *   Example: `data/medqa_usmle.jsonl`
*   `--subset_1000` (Optional, flag): If provided, the system will only process the first 1000 questions from the dataset.
    *   Default: Processes all questions.
*   `--max_wiki_pages` (Optional, integer): Maximum number of Wikipedia pages to retrieve and process for each question.
    *   Default: `5`
*   `--faiss_top_k` (Optional, integer): The number of top relevant evidence sentences/segments to retrieve from the FAISS index for each question option during scoring.
    *   Default: `1` (as per the `evidence_scorer.py` default) but `main.py` default is currently also 1. *The default in `main.py` is 5, this was updated later.*

### Example Usage:

1.  **Run with a specific dataset file (processing all questions):**
    ```bash
    python main.py data/your_medqa_file.jsonl
    ```

2.  **Run with a subset of 1000 questions:**
    ```bash
    python main.py data/your_medqa_file.jsonl --subset_1000
    ```

3.  **Specify the number of Wikipedia pages and FAISS top K evidence segments:**
    ```bash
    python main.py data/your_medqa_file.jsonl --max_wiki_pages 3 --faiss_top_k 3
    ```

## Output

*   The system logs its progress extensively to the console. Log messages include information about data loading, model initialization, processing steps for each question, and any warnings or errors encountered.
*   For each question, it logs:
    *   The question ID (if available, otherwise an index).
    *   The scores calculated for each option.
    *   The predicted answer key and the correct answer key.
    *   Whether the prediction was CORRECT or INCORRECT.
*   At the end of the run, the system prints and logs the overall accuracy:
    `Overall Accuracy: X.XXXX (CorrectPredictions/TotalEvaluatedQuestions)`

## Running Tests

Unit tests are provided in the `tests/` directory to verify the functionality of individual modules.

To run all tests:
```bash
python -m unittest discover -s tests -p "test_*.py"
```
This command will discover and run all files in the `tests` directory that start with `test_` and end with `.py`.

## Placeholders / Future Work

*   **Advanced Models**: The modules `src/distilbert_classifier.py` and `src/distilbert_nli.py` are currently placeholders. Future work could involve implementing these to:
    *   Fine-tune a DistilBERT model for question categorization (e.g., to adapt the retrieval strategy).
    *   Fine-tune a DistilBERT model for Natural Language Inference (NLI) to score the entailment between retrieved evidence and question-option pairs, potentially providing a more robust scoring mechanism.
*   **Error Analysis**: More detailed error analysis and logging of specific failure cases for incorrect predictions.
*   **Knowledge Source Expansion**: Integrating other knowledge sources beyond Wikipedia.
*   **Hybrid Retrieval**: Combining sparse retrieval (e.g., TF-IDF/BM25) with dense retrieval (FAISS) for candidate evidence selection.
*   **Answer Justification**: Extracting or generating justifications for the predicted answers.
*   **Configuration File**: Moving model names, default parameters, etc., to a configuration file.
```
