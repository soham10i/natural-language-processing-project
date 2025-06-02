import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
import nltk # Keep for sent_tokenize and potential download
import faiss
# import re # Not currently used for fallback splitting, using string.split

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download nltk.punkt if not already available
# This is a module-level side effect, generally okay for widely used resources like punkt.
# Consider moving to an explicit setup script or a one-time check in main.py if preferred.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    logger.info("NLTK 'punkt' tokenizer not found. Attempting to download...")
    try:
        nltk.download('punkt', quiet=True)
        logger.info("'punkt' tokenizer downloaded successfully.")
    except Exception as e_nltk_download:
        logger.error(f"Failed to download 'punkt' tokenizer: {e_nltk_download}. sent_tokenize may not work.", exc_info=True)
        # Application might fail later if sent_tokenize is crucial and unavailable.
from nltk.tokenize import sent_tokenize


def score_options_similarity_with_vector_db(processed_question, wikipedia_texts, sentence_model, device, top_k_evidence=1):
    """
    Scores question options against Wikipedia text segments using FAISS for similarity search.

    Parameters
    ----------
    processed_question : dict
        A dictionary from `question_processor.process_question`, containing at least:
            - 'original_question_data': dict with 'question' (str), 'options' (dict of str), and 'id' (str).
    wikipedia_texts : list of str
        A list of strings, where each string is the text content of a Wikipedia page.
    sentence_model : sentence_transformers.SentenceTransformer
        A loaded SentenceTransformer model.
    device : torch.device
        The PyTorch device (e.g., "cuda" or "cpu").
    top_k_evidence : int, optional
        The number of top evidence sentences to retrieve for each option. Default is 1.
        The score returned is the maximum similarity among these top_k sentences.

    Returns
    -------
    dict
        A dictionary mapping option keys (e.g., 'A', 'B') to their similarity scores.
        Returns empty dict if errors occur or no segments are found.
    """
    if not processed_question or 'original_question_data' not in processed_question:
        logging.error("Invalid `processed_question` input.")
        return {}
    if not wikipedia_texts:
        logging.warning("No Wikipedia texts provided for scoring.")
        return {key: 0.0 for key in processed_question['original_question_data'].get('options', {}).keys()}

    # Text Segmentation
    full_text = "\n\n".join(wikipedia_texts)
    segments = sent_tokenize(full_text)
    segments = [s.strip() for s in segments if len(s.strip().split()) >= 5] # Filter short sentences

    if not segments:
        logging.warning("No usable sentences after tokenization. Falling back to paragraph splitting.")
        # Fallback 1: split by double newline (paragraphs)
        segments = [s.strip() for s in full_text.split('\n\n') if len(s.strip().split()) >= 5]
        if not segments:
            logging.warning("Fallback to paragraph splitting yielded no segments. Using raw wikipedia_texts.")
            # Fallback 2: use raw texts if they are substantial enough
            segments = [s.strip() for s in wikipedia_texts if len(s.strip().split()) >= 5]
            if not segments:
                logging.error("No usable text segments found in Wikipedia texts.")
                return {key: 0.0 for key in processed_question['original_question_data'].get('options', {}).keys()}
    
    logging.info(f"Generated {len(segments)} text segments for FAISS indexing.")

    # Segment Embedding
    try:
        segment_embeddings = sentence_model.encode(segments, device=device, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
        if segment_embeddings.ndim == 1: # Should be 2D
            segment_embeddings = segment_embeddings.unsqueeze(0)
        
        # Normalization for cosine similarity (IndexFlatIP expects normalized vectors for cosine)
        segment_embeddings_normalized = segment_embeddings / segment_embeddings.norm(dim=1, keepdim=True)
        segment_embeddings_np = segment_embeddings_normalized.cpu().numpy()
    except Exception as e:
        logging.error(f"Error generating segment embeddings: {e}")
        return {key: 0.0 for key in processed_question['original_question_data'].get('options', {}).keys()}

    if segment_embeddings_np.shape[0] == 0:
        logging.error("Segment embeddings are empty.")
        return {key: 0.0 for key in processed_question['original_question_data'].get('options', {}).keys()}

    # FAISS Integration
    try:
        embedding_dim = segment_embeddings_np.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)

        # GPU acceleration if available and faiss-gpu is installed
        if device.type == 'cuda':
            try:
                gpu_id = int(device.index) if device.index is not None else 0
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, gpu_id, index)
                logging.info(f"Using FAISS on GPU: {gpu_id}")
            except AttributeError: # faiss might be CPU-only version
                logging.info("FAISS GPU resources not available or faiss-gpu not installed. Using CPU.")
            except Exception as e_gpu: # Other potential errors with GPU setup
                logging.warning(f"Failed to move FAISS index to GPU: {e_gpu}. Using CPU.")
        
        index.add(segment_embeddings_np)
    except Exception as e:
        logging.error(f"Error initializing or adding to FAISS index: {e}")
        return {key: 0.0 for key in processed_question['original_question_data'].get('options', {}).keys()}

    # Option Scoring
    option_scores = {}
    original_question_text = processed_question['original_question_data']['question']
    options_data = processed_question['original_question_data'].get('options', {})

    if not options_data:
        logging.warning("No options found in processed_question to score.")
        return {}

    for option_key, option_text in options_data.items():
        if not isinstance(option_text, str):
            logging.warning(f"Option '{option_key}' text is not a string: {option_text}. Assigning score 0.")
            option_scores[option_key] = 0.0
            continue

        combined_query = f"{original_question_text} <SEP> {option_text}"
        try:
            query_embedding = sentence_model.encode(combined_query, device=device, convert_to_tensor=True, show_progress_bar=False)
            if query_embedding.ndim == 1: # Ensure 2D for norm and FAISS
                 query_embedding_normalized = query_embedding / query_embedding.norm()
                 query_embedding_normalized = query_embedding_normalized.unsqueeze(0)
            else: # Should not happen with single query string
                 query_embedding_normalized = query_embedding / query_embedding.norm(dim=1, keepdim=True)

            query_embedding_np = query_embedding_normalized.cpu().numpy()

            # Search FAISS
            D, I = index.search(query_embedding_np, k=min(top_k_evidence, index.ntotal)) # D=distances (similarities), I=indices
            
            if D is not None and D.size > 0:
                # Score is the max similarity among the top_k retrieved sentences
                score = np.max(D[0]) if D[0].size > 0 else 0.0
            else:
                score = 0.0 # No results or error
            
            option_scores[option_key] = float(score)
            logging.debug(f"Option '{option_key}', Query: '{combined_query[:100]}...', Score: {score:.4f}")

        except Exception as e:
            logging.error(f"Error scoring option '{option_key}': {e}")
            option_scores[option_key] = 0.0

    return option_scores

if __name__ == '__main__':
    logging.info("evidence_scorer.py executed directly. This script is intended to be imported as a module.")
    # Example usage (requires models, data, and FAISS installation):
    # from sentence_transformers import SentenceTransformer
    # dummy_processed_question = {
    #     'original_question_data': {
    #         'question': "What is the primary treatment for type 1 diabetes?",
    #         'options': {
    #             'A': "Metformin",
    #             'B': "Insulin",
    #             'C': "Dietary changes",
    #             'D': "Oral hypoglycemics"
    #         }
    #     }
    # }
    # dummy_wiki_texts = [
    #     "Insulin is essential for managing type 1 diabetes. It helps regulate blood sugar.",
    #     "Type 1 diabetes mellitus is an autoimmune condition. The pancreas produces little to no insulin.",
    #     "Metformin is typically used for type 2 diabetes, not type 1.",
    #     "While diet is important, it cannot replace insulin for type 1 diabetes.",
    #     "Some oral medications are used for type 2 diabetes."
    # ]
    # try:
    #     sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    #     current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     sbert_model.to(current_device)
        
    #     scores = score_options_similarity_with_vector_db(dummy_processed_question, dummy_wiki_texts, sbert_model, current_device)
    #     if scores:
    #         print("Option Scores (dummy example):")
    #         for option, score in scores.items():
    #             print(f"  {option}: {score:.4f}")
    #     else:
    #         print("No scores returned in dummy example.")
            
    # except ImportError:
    #     print("FAISS or SentenceTransformers not installed. Cannot run example.")
    # except Exception as e:
    #     print(f"Could not run example due to: {e}.")
