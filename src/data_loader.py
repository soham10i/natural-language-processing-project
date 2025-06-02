import json
import logging

# Configure basic logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# BasicConfig should be in main.py only. Modules use getLogger.
logger = logging.getLogger(__name__)

def load_medqa_data(file_path):
    """Load MedQA data from a JSONL file.

    Parameters
    ----------
    file_path : str
        The path to the JSONL file containing MedQA data.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary represents a data sample.
        Returns an empty list if the file is not found or an error occurs during parsing.
    """
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON from line: {line.strip()} - {e}")
        logging.info(f"Successfully loaded data from {file_path}. Number of samples: {len(data)}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading data from {file_path}: {e}")
        return []

if __name__ == '__main__':
    # Example usage (optional, for testing purposes)
    # Create a dummy data.jsonl file for testing
    dummy_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Hamlet?", "answer": "Shakespeare"}
    ]
    dummy_file_path = "dummy_data.jsonl"
    with open(dummy_file_path, 'w') as f:
        for item in dummy_data:
            f.write(json.dumps(item) + '\\n')

    loaded_data = load_medqa_data(dummy_file_path)
    if loaded_data:
        print("Data loaded successfully (dummy data):")
        for item in loaded_data:
            print(item)

    # Test FileNotFoundError
    load_medqa_data("non_existent_file.jsonl")
