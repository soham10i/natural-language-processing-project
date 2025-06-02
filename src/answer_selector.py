import logging

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def select_answer(option_scores):
    """
    Selects the answer option with the highest score.

    Parameters
    ----------
    option_scores : dict
        A dictionary where keys are option identifiers (e.g., 'A', 'B')
        and values are their corresponding scores (float).
        Example: {'A': 0.8, 'B': 0.9, 'C': 0.75}

    Returns
    -------
    str or None
        The option key with the highest score (e.g., 'B').
        Returns None if option_scores is empty, not a dictionary, or an error occurs.
    """
    logger.debug(f"Attempting to select answer from scores: {option_scores}")
    if not isinstance(option_scores, dict) or not option_scores:
        logger.warning(f"`option_scores` is empty or not a valid dictionary: {option_scores}. Cannot select an answer.")
        return None

    try:
        # Find the option key with the maximum score
        selected_key = max(option_scores, key=option_scores.get)
        logger.info(f"Selected answer: '{selected_key}' with score {option_scores[selected_key]:.4f}")
        return selected_key
    except ValueError: # max() arg is an empty sequence
        logger.warning(f"ValueError (likely empty option_scores though caught by initial check) for scores: {option_scores}. Cannot select answer.", exc_info=True)
        return None
    except Exception as e: # Catch any other unexpected error
        logger.error(f"An unexpected error occurred while selecting the answer from scores {option_scores}: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # BasicConfig for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("answer_selector.py executed directly for testing purposes.")
    
    # Example Usage
    scores_example1 = {'A': 0.8, 'B': 0.9, 'C': 0.75}
    logger.info(f"Scores: {scores_example1}, Selected: {select_answer(scores_example1)}") 

    scores_example2 = {'X': 0.1, 'Y': 0.05, 'Z': 0.12}
    logger.info(f"Scores: {scores_example2}, Selected: {select_answer(scores_example2)}") 

    scores_example3 = {}
    logger.info(f"Scores: {scores_example3}, Selected: {select_answer(scores_example3)}") 

    scores_example4 = {'A': -0.5, 'B': -0.1, 'C': -0.8}
    logger.info(f"Scores: {scores_example4}, Selected: {select_answer(scores_example4)}") 
    
    scores_example5 = None # Test invalid input type
    logger.info(f"Scores: {scores_example5}, Selected: {select_answer(scores_example5)}")
    
    scores_example6 = {"OnlyOne": 1.0}
    logger.info(f"Scores: {scores_example6}, Selected: {select_answer(scores_example6)}")
