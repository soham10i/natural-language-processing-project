import unittest
import json
import os
import logging
from unittest.mock import patch, mock_open # For mocking open and logging

# Adjust import path based on your project structure.
# If tests/ is a top-level sibling to src/, this should work.
# If your runner is in the root and runs `python -m unittest discover tests`,
# then `from src.data_loader import load_medqa_data` is correct.
from src.data_loader import load_medqa_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up for test methods."""
        self.test_file_path = "test_data.jsonl"
        self.sample_q1 = {"id": "q1", "question": "What is q1?", "answer": "a1", "options": {"A": "oA", "B": "oB"}, "answer_idx": "A", "metamap_phrases": ["m1"]}
        self.sample_q2 = {"id": "q2", "question": "Explain q2.", "answer": "a2", "options": {"A": "oX", "B": "oY"}, "answer_idx": "B", "metamap_phrases": ["m2", "m3"]}

    def tearDown(self):
        """Tear down after test methods."""
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)

    def test_successful_load(self):
        """Test loading a valid JSONL file."""
        with open(self.test_file_path, 'w') as f:
            f.write(json.dumps(self.sample_q1) + '\n')
            f.write(json.dumps(self.sample_q2) + '\n')

        loaded_data = load_medqa_data(self.test_file_path)
        self.assertIsNotNone(loaded_data)
        self.assertEqual(len(loaded_data), 2)
        self.assertEqual(loaded_data[0]['question'], self.sample_q1['question'])
        self.assertEqual(loaded_data[1]['options'], self.sample_q2['options'])
        self.assertEqual(loaded_data[0]['metamap_phrases'], self.sample_q1['metamap_phrases'])


    @patch('src.data_loader.logging.error') # Mock logging.error within the data_loader module
    def test_file_not_found(self, mock_log_error):
        """Test loading a non-existent file."""
        non_existent_file = "no_such_file.jsonl"
        # load_medqa_data returns [] for FileNotFoundError
        result = load_medqa_data(non_existent_file)
        self.assertEqual(result, [])
        mock_log_error.assert_called_with(f"File not found: {non_existent_file}")

    @patch('src.data_loader.logging.error') # Mock logging.error
    def test_invalid_json_format(self, mock_log_error):
        """Test loading a file with invalid JSON content."""
        with open(self.test_file_path, 'w') as f:
            f.write("this is not valid json\n")
            f.write(json.dumps(self.sample_q1) + '\n') # A valid line after an invalid one

        loaded_data = load_medqa_data(self.test_file_path)
        # The current implementation of load_medqa_data tries to load line by line
        # and logs an error for the bad line, then continues.
        self.assertEqual(len(loaded_data), 1) # Only the valid line should be loaded
        self.assertEqual(loaded_data[0]['question'], self.sample_q1['question'])
        # Check that an error was logged for the invalid line
        mock_log_error.assert_any_call(f"Error decoding JSON from line: this is not valid json - Expecting value: line 1 column 1 (char 0)")


    @patch('src.data_loader.logging.error')
    def test_empty_file(self, mock_log_error):
        """Test loading an empty file."""
        with open(self.test_file_path, 'w') as f:
            pass # Create an empty file
        
        loaded_data = load_medqa_data(self.test_file_path)
        self.assertEqual(loaded_data, [])
        # No error should be logged for an empty file, it's a valid (though empty) case.
        mock_log_error.assert_not_called()

    @patch('src.data_loader.logging.error')
    def test_partially_valid_json(self, mock_log_error):
        """Test a file with some valid and some invalid JSON lines."""
        with open(self.test_file_path, 'w') as f:
            f.write(json.dumps(self.sample_q1) + '\n')
            f.write("invalid json line here\n")
            f.write(json.dumps(self.sample_q2) + '\n')
            f.write("{'malformed': json,}\n") # another invalid json

        loaded_data = load_medqa_data(self.test_file_path)
        self.assertEqual(len(loaded_data), 2) # Should load the two valid JSON objects
        self.assertEqual(loaded_data[0]['id'], 'q1')
        self.assertEqual(loaded_data[1]['id'], 'q2')
        
        # Check that errors were logged for the invalid lines
        self.assertEqual(mock_log_error.call_count, 2)
        mock_log_error.assert_any_call("Error decoding JSON from line: invalid json line here - Expecting value: line 1 column 1 (char 0)")
        mock_log_error.assert_any_call("Error decoding JSON from line: {'malformed': json,} - Expecting property name enclosed in double quotes: line 1 column 2 (char 1)")


if __name__ == '__main__':
    unittest.main()
