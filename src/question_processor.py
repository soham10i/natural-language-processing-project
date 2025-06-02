import spacy
from sentence_transformers import SentenceTransformer
import torch
import logging

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_question(question_data, nlp_spacy, sentence_model, device):
    """
    Processes a single question data dictionary to extract entities and generate embeddings.

    Parameters
    ----------
    question_data : dict
        A dictionary containing the question, options, and potentially metamap_phrases.
        Expected keys: 'question' (str), 'options' (dict of str), 'metamap_phrases' (list of str, optional).
    nlp_spacy : spacy.language.Language
        A loaded SciSpacy model for Named Entity Recognition (NER).
    sentence_model : sentence_transformers.SentenceTransformer
        A loaded SentenceTransformer model for generating sentence embeddings.
        This model is expected to be already on the correct `device`.
    device : torch.device
        The PyTorch device (e.g., "cuda" or "cpu") for tensor operations.

    Returns
    -------
    dict or None
        A dictionary containing:
            - 'original_question_data': The input question_data.
            - 'extracted_entities': A list of unique entities extracted from the question and options.
            - 'question_embedding': A PyTorch tensor representing the sentence embedding of the question.
        Returns None if essential keys like 'question' or 'options' are missing.
    """
    question_id = question_data.get('id', 'N/A') # For more informative logging
    logger.debug(f"Processing question ID: {question_id}")

    if not all(k in question_data for k in ['question', 'options']):
        logger.error(f"Missing 'question' or 'options' in question_data for ID {question_id}. Cannot process.")
        return None

    # Extract Metamap phrases
    metamap_phrases = question_data.get('metamap_phrases', [])
    if not metamap_phrases:
        logger.debug(f"No metamap_phrases found or provided in question_data for ID {question_id}.")
    else:
        logger.debug(f"Metamap phrases for ID {question_id}: {metamap_phrases}")
    extracted_entities_set = set(metamap_phrases)


    # NER on the question itself
    question_text = question_data.get('question')
    if not question_text or not isinstance(question_text, str):
        logger.warning(f"Question text for ID {question_id} is empty, missing, or not a string. Skipping NER on question.")
        cleaned_question = "" # Fallback for cleaned_question if question is missing/invalid
    else:
        logger.debug(f"Original question (ID {question_id}): {question_text}")
        cleaned_question = question_text.lower() # Clean question text (simple lowercase for now)
        try:
            question_doc = nlp_spacy(question_text)
            for ent in question_doc.ents:
                extracted_entities_set.add(ent.text)
                logger.debug(f"NER from question (ID {question_id}): '{ent.text}'")
        except Exception as e:
            logger.error(f"Error during NER processing of question for ID {question_id}: {e}", exc_info=True)


    # NER on each option's value
    options_data = question_data.get('options')
    if isinstance(options_data, dict):
        for option_key, option_text in options_data.items():
            if isinstance(option_text, str):
                if not option_text.strip():
                    logger.debug(f"Option '{option_key}' for question ID {question_id} is an empty string, skipping NER.")
                    continue
                try:
                    option_doc = nlp_spacy(option_text)
                    for ent in option_doc.ents:
                        extracted_entities_set.add(ent.text)
                        logger.debug(f"NER from option '{option_key}' (ID {question_id}): '{ent.text}'")
                except Exception as e:
                    logger.error(f"Error during NER processing of option '{option_key}' for ID {question_id}: {e}", exc_info=True)
            else:
                logger.warning(f"Option '{option_key}' for question ID {question_id} text is not a string (type: '{type(option_text)}'), skipping NER for this option.")
    else:
        logger.warning(f"Options for question ID {question_id} are not a dictionary or missing, skipping NER on options.")


    list_of_unique_entities = list(extracted_entities_set)
    logger.debug(f"Extracted {len(list_of_unique_entities)} unique entities for ID {question_id}: {list_of_unique_entities}")


    # Generate sentence embedding for the cleaned question
    if not cleaned_question.strip():
        logger.warning(f"Cleaned question text is empty for question ID {question_id}. Embedding will be for an empty string.")
        # SentenceTransformer typically handles empty strings, producing some consistent embedding.
        
    try:
        # Pass show_progress_bar=False if not needed, to avoid potential tqdm overhead in logs
        question_embedding = sentence_model.encode(cleaned_question, device=device, convert_to_tensor=True, show_progress_bar=False)
        logger.debug(f"Generated question embedding of shape {question_embedding.shape} for ID {question_id}")
    except Exception as e:
        logger.error(f"Error generating sentence embedding for question ID {question_id}: {e}", exc_info=True)
        # Fallback: return None or a zero tensor if critical, or let error propagate
        # For this structure, if encode fails and doesn't return a tensor, the dict might be ill-formed.
        # Let's assume for now the main processing loop will catch this if it's critical.
        # A possible safe fallback if an embedding MUST be returned:
        # question_embedding = torch.zeros(sentence_model.get_sentence_embedding_dimension(), device=device) 
        # However, this requires knowing the dimension. For now, we'll let it be as is.
        # The impact of a failed embedding should be handled by the caller or main loop.
        # If encode returns None or raises, the dict creation below might fail or contain None.
        # The function signature says it returns a dict, so it should ideally always do so or return None if it can't fulfill the contract.
        # Given the current structure, if encode fails and raises, the function execution stops.
        # If it returns None, then 'question_embedding': None will be in the dict.
        # Let's assume process_question should return None if embedding generation fails critically.
        return None # Or handle more gracefully if a partial result is acceptable


    # Placeholder for question categorization logic
    # TODO: Implement question categorization (e.g., based on keywords, structure, etc.)
    logger.debug(f"Finished processing question ID: {question_id}")

    return {
        'original_question_data': question_data,
        'extracted_entities': list_of_unique_entities,
        'question_embedding': question_embedding
    }

if __name__ == '__main__':
    # This is a placeholder for example usage or testing.
    # For actual testing, you would need to:
    # 1. Load a spacy model: nlp = spacy.load("en_core_sci_sm") or other scispaCy model
    # 2. Load a sentence transformer model: model = SentenceTransformer('all-MiniLM-L6-v2')
    # 3. Set up a device: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 4. Create sample question_data
    
    # BasicConfig for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("question_processor.py executed directly for testing purposes.")
    
    dummy_question_example = {
        "id": "dummy_qp1",
        "question": "What is the treatment for Type 2 Diabetes Mellitus?",
        "options": {"A": "Insulin therapy", "B": "Metformin medication", "C": "Dietary changes", "D": "Physical exercise"},
        "metamap_phrases": ["Type 2 Diabetes Mellitus", "Treatment"]
    }
    try:
        logger.info("--- Running standalone example for question_processor ---")
        
        # Attempt to load models for the example. These are heavy dependencies for a simple module test.
        # In a real testing scenario for this module, these would be mocked (as in test_question_processor.py)
        try:
            nlp_example = spacy.load("en_core_web_sm") # Using a smaller, more common model for example
            logger.info("Loaded spacy 'en_core_web_sm' for example.")
        except OSError:
            logger.error("Spacy 'en_core_web_sm' not found. Please download it: python -m spacy download en_core_web_sm")
            nlp_example = None

        try:
            model_example = SentenceTransformer('all-MiniLM-L6-v2') # A common SBERT model
            logger.info("Loaded SentenceTransformer 'all-MiniLM-L6-v2' for example.")
        except Exception as e_sbert:
            logger.error(f"Could not load SentenceTransformer 'all-MiniLM-L6-v2': {e_sbert}")
            model_example = None

        current_device_example = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_example:
            model_example.to(current_device_example)
            logger.debug(f"Test models loaded on {current_device_example}")
        
        if nlp_example and model_example:
            processed_q_example = process_question(dummy_question_example, nlp_example, model_example, current_device_example)
            
            if processed_q_example:
                logger.info("Processed Question (Example):")
                logger.info(f"  Original ID: {processed_q_example['original_question_data'].get('id')}")
                logger.info(f"  Original Question: {processed_q_example['original_question_data']['question']}")
                logger.info(f"  Entities: {processed_q_example['extracted_entities']}")
                if 'question_embedding' in processed_q_example and processed_q_example['question_embedding'] is not None:
                     logger.info(f"  Embedding shape: {processed_q_example['question_embedding'].shape}")
                else:
                     logger.warning("  Question embedding is missing or None in the result.")
            else:
                logger.error("Processing failed for the example question.")
        else:
            logger.warning("One or more models (spaCy, SentenceTransformer) could not be loaded. Skipping example run.")
            
    except Exception as e_main_test:
        logger.error(f"Could not run standalone example for question_processor due to: {e_main_test}", exc_info=True)
