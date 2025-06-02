import wikipedia
import torch
import logging
import time
# import requests # Not currently used
# from bs4 import BeautifulSoup # Not currently used

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_wikipedia_pages(processed_question, sentence_model, device, max_pages=5):
    """
    Selects relevant Wikipedia page contents based on entities extracted from a processed question.

    Parameters
    ----------
    processed_question : dict
        A dictionary from `question_processor.process_question`, containing:
            - 'extracted_entities': A list of unique entities.
            - 'question_embedding': A PyTorch tensor for the question embedding.
            - 'original_question_data': dict, containing at least 'id' for logging.
    sentence_model : sentence_transformers.SentenceTransformer
        A loaded SentenceTransformer model for generating sentence embeddings.
        This model is expected to be on the correct `device`.
    device : torch.device
        The PyTorch device (e.g., "cuda" or "cpu") for tensor operations.
    max_pages : int, optional
        The maximum number of Wikipedia page contents to return, by default 5.

    Returns
    -------
    list of str
        A list of strings, where each string is the text content of a selected Wikipedia page.
        Returns an empty list if no relevant pages are found or errors occur.
    """
    question_id = processed_question.get('original_question_data', {}).get('id', 'N/A_ws') # For logging
    logger.debug(f"Starting Wikipedia page selection for question ID: {question_id}")

    if not processed_question or 'extracted_entities' not in processed_question or 'question_embedding' not in processed_question:
        logger.error(f"Invalid `processed_question` input for ID {question_id}. Missing keys like 'extracted_entities' or 'question_embedding'.")
        return []

    entities = processed_question['extracted_entities']
    question_embedding = processed_question['question_embedding']


    if not entities:
        logger.info(f"No entities extracted for question ID {question_id}, cannot search Wikipedia.")
        return []

    # Query Generation: Select the first 3-5 entities
    num_queries = min(max(1, len(entities)), 5) 
    search_queries = entities[:num_queries]
    logger.debug(f"Using queries for Wikipedia search (ID {question_id}): {search_queries}")

    # Candidate Retrieval
    candidate_page_titles = set()
    for query in search_queries:
        try:
            logger.debug(f"Searching Wikipedia for query: '{query}' (ID: {question_id})")
            search_results = wikipedia.search(query, results=5) # Get top 5 suggestions for the query
            for title in search_results:
                candidate_page_titles.add(title)
            time.sleep(0.05) # Small delay to be respectful to Wikipedia API
        except wikipedia.exceptions.DisambiguationError as e:
            logger.warning(f"Disambiguation error for query '{query}' (ID: {question_id}): {e}. Trying first suggestion if available.", exc_info=True)
            if e.options: # DisambiguationError has an 'options' attribute with suggestions
                candidate_page_titles.add(e.options[0])
        except Exception as e: # Catch other potential wikipedia library errors
            logger.error(f"Error searching Wikipedia for query '{query}' (ID: {question_id}): {e}", exc_info=True)
    
    if not candidate_page_titles:
        logger.info(f"No candidate Wikipedia pages found from search for question ID: {question_id}")
        return []
    
    logger.debug(f"Found {len(candidate_page_titles)} unique candidate page titles for ID {question_id}: {candidate_page_titles}")

    # Page Ranking & Selection
    ranked_pages = []
    # Ensure question_embedding is 2D for cosine_similarity: [1, dim]
    question_embedding_2d = question_embedding.unsqueeze(0) if question_embedding.ndim == 1 else question_embedding

    for title in candidate_page_titles:
        page_summary = None
        try:
            logger.debug(f"Fetching summary for page '{title}' (ID: {question_id})")
            # wikipedia.summary can sometimes raise DisambiguationError or PageError
            page_summary = wikipedia.summary(title, sentences=5) # Fetch a few sentences for summary
            time.sleep(0.05) # Respectful delay

            if not page_summary: 
                logger.warning(f"Empty summary for page '{title}' (ID: {question_id}), skipping similarity calculation.")
                continue

            # Encode summary; ensure show_progress_bar is False for cleaner logs unless debugging that part
            summary_embedding = sentence_model.encode(page_summary, device=device, convert_to_tensor=True, show_progress_bar=False)
            summary_embedding_2d = summary_embedding.unsqueeze(0) if summary_embedding.ndim == 1 else summary_embedding
            
            # Ensure embeddings are on the same device (usually handled by sentence_model.to(device) and passing device to encode)
            if question_embedding_2d.device != summary_embedding_2d.device:
                 logger.warning(f"Question embedding on {question_embedding_2d.device} and summary for '{title}' on {summary_embedding_2d.device}. Moving summary to question's device for ID {question_id}.")
                 summary_embedding_2d = summary_embedding_2d.to(question_embedding_2d.device)

            similarity_score_tensor = torch.nn.functional.cosine_similarity(question_embedding_2d, summary_embedding_2d)
            similarity_score = similarity_score_tensor.item() # Get scalar value
            
            ranked_pages.append((title, similarity_score, page_summary)) # Store summary for context if needed later
            logger.debug(f"Page '{title}' (ID: {question_id}), Similarity: {similarity_score:.4f}")

        except wikipedia.exceptions.PageError: # Specific page does not exist
            logger.warning(f"Wikipedia PageError for title '{title}' (ID: {question_id}). Page may not exist. Skipping.", exc_info=True)
        except wikipedia.exceptions.DisambiguationError as e: # Page is a disambiguation page
            logger.warning(f"Wikipedia DisambiguationError for title '{title}' during summary fetch (ID: {question_id}): {e}. Skipping.", exc_info=True)
        except Exception as e: # Catch other errors during summary processing or embedding
            logger.error(f"Unexpected error processing page '{title}' for summary/ranking (ID: {question_id}): {e}", exc_info=True)
        
        # time.sleep(0.05) # Delay after each API call, already have one after summary

    if not ranked_pages:
        logger.info(f"No pages could be ranked (e.g., all summaries failed) for question ID: {question_id}")
        return []

    # Sort pages by similarity score in descending order
    ranked_pages.sort(key=lambda x: x[1], reverse=True)
    logger.debug(f"Ranked pages for ID {question_id} (top 5): {[(p[0], p[1]) for p in ranked_pages[:5]]}")


    # Select top N unique page titles (already unique from set, but ranking might reintroduce via different search terms mapping to same canonical)
    selected_page_titles = []
    seen_titles_for_selection = set() # Ensure final list is unique if multiple queries led to same ranked page
    for r_title, r_score, _ in ranked_pages: # Iterate through ranked pages
        if r_title not in seen_titles_for_selection:
            selected_page_titles.append(r_title)
            seen_titles_for_selection.add(r_title)
            if len(selected_page_titles) >= max_pages:
                break
    
    logger.info(f"Selected top {len(selected_page_titles)} pages for content fetching (ID {question_id}): {selected_page_titles}")

    # Content Fetching
    page_contents = []
    for content_title in selected_page_titles:
        try:
            logger.debug(f"Fetching full content for page: '{content_title}' (ID: {question_id})")
            page_obj = wikipedia.page(content_title) # Can raise PageError or DisambiguationError
            page_contents.append(page_obj.content)
            logger.debug(f"Successfully fetched and stored content for page: '{content_title}' (ID: {question_id})")
        except wikipedia.exceptions.PageError:
            logger.warning(f"Wikipedia PageError fetching full content for '{content_title}' (ID: {question_id}). Skipping.", exc_info=True)
        except wikipedia.exceptions.DisambiguationError as e: # Should ideally not happen if summaries were fetched for these titles
            logger.warning(f"Wikipedia DisambiguationError fetching full content for '{content_title}' (ID: {question_id}): {e}. Skipping.", exc_info=True)
        except Exception as e: # Other errors like connection issues
            logger.error(f"Error fetching full content for page '{content_title}' (ID: {question_id}): {e}", exc_info=True)
        time.sleep(0.05) # Respectful delay after each full page load

    logger.info(f"Fetched content for {len(page_contents)} pages for question ID: {question_id}.")
    return page_contents

if __name__ == '__main__':
    # This is a placeholder for example usage or testing.
    # For actual testing, you would need:
    # 1. A SentenceTransformer model and device.
    # 2. A sample `processed_question` dictionary.
    
    # BasicConfig for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("wikipedia_selector.py executed directly for testing purposes.")
    
    from sentence_transformers import SentenceTransformer # Import here for standalone test
    
    dummy_processed_question_example = {
        'extracted_entities': ["Type 2 Diabetes Mellitus", "Metformin", "Treatment Guidelines for Diabetes"],
        # Simulating a 384-dim embedding, as from all-MiniLM-L6-v2
        'question_embedding': torch.rand((384,)), 
        'original_question_data': {'id': 'dummy_ws1'} # Added for context
    }
    try:
        logger.info("--- Running standalone example for wikipedia_selector ---")
        
        sbert_model_example = SentenceTransformer('all-MiniLM-L6-v2')
        current_device_example = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sbert_model_example.to(current_device_example)
        
        # Ensure question embedding is on the same device as the model for the example
        if dummy_processed_question_example['question_embedding'].device != current_device_example:
           dummy_processed_question_example['question_embedding'] = dummy_processed_question_example['question_embedding'].to(current_device_example)
        
        logger.debug(f"Test models loaded on {current_device_example} for wikipedia_selector example.")
        
        pages_example = select_wikipedia_pages(dummy_processed_question_example, sbert_model_example, current_device_example, max_pages=2)
        if pages_example:
            logger.info(f"Retrieved {len(pages_example)} page contents (Example):")
            for i_example, content_example in enumerate(pages_example):
                logger.debug(f"--- Page {i_example+1} (first 200 chars) ---")
                logger.debug(content_example[:200].replace('\n', ' ') + "...") # Replace newlines for cleaner log
        else:
            logger.info("No pages retrieved in wikipedia_selector example.")
            
    except Exception as e_test_main:
        logger.error(f"Could not run wikipedia_selector example due to: {e_test_main}", exc_info=True)
