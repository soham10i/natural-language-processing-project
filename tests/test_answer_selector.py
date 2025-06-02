import unittest
import logging
from unittest.mock import patch

# Adjust import path based on your project structure
from src.answer_selector import select_answer

class TestAnswerSelector(unittest.TestCase):

    def test_basic_selection(self):
        """Test basic functionality: selecting the key with the highest score."""
        option_scores = {'A': 0.1, 'B': 0.9, 'C': 0.5}
        self.assertEqual(select_answer(option_scores), 'B')

    def test_another_basic_selection(self):
        """Test with different scores."""
        option_scores = {'X': 0.85, 'Y': 0.75, 'Z': 0.92}
        self.assertEqual(select_answer(option_scores), 'Z')

    def test_tie_breaking_implicit(self):
        """Test tie-breaking (current implementation likely picks first or by dict order)."""
        option_scores = {'A': 0.9, 'B': 0.9, 'C': 0.5}
        # The exact behavior for ties depends on `max(dict, key=dict.get)`
        # which typically returns the first key encountered with the max value.
        # This can be unpredictable if dict order changes, but for testing,
        # we can assert it's one of the tied keys.
        result = select_answer(option_scores)
        self.assertIn(result, ['A', 'B']) 

    def test_empty_scores_dictionary(self):
        """Test with an empty scores dictionary."""
        option_scores = {}
        # Based on implementation, select_answer logs a warning and returns None
        with patch.object(logging, 'warning') as mock_log_warning:
            self.assertIsNone(select_answer(option_scores))
            mock_log_warning.assert_called_with("`option_scores` is empty or not a dictionary. Cannot select an answer.")

    def test_scores_with_negative_values(self):
        """Test with negative score values."""
        option_scores = {'A': -0.5, 'B': -0.1, 'C': -0.8}
        self.assertEqual(select_answer(option_scores), 'B')

    def test_single_option(self):
        """Test with only one option."""
        option_scores = {'M': 0.99}
        self.assertEqual(select_answer(option_scores), 'M')
        
    def test_all_scores_zero(self):
        """Test when all scores are zero."""
        option_scores = {'A': 0.0, 'B': 0.0, 'C': 0.0}
        result = select_answer(option_scores)
        self.assertIn(result, ['A', 'B', 'C']) # Similar to tie-breaking

    def test_non_dictionary_input(self):
        """Test with input that is not a dictionary."""
        with patch.object(logging, 'warning') as mock_log_warning:
            self.assertIsNone(select_answer(None)) # Test with None
            mock_log_warning.assert_called_with("`option_scores` is empty or not a dictionary. Cannot select an answer.")
        
        with patch.object(logging, 'warning') as mock_log_warning:
            self.assertIsNone(select_answer("not_a_dict")) # Test with a string
            mock_log_warning.assert_called_with("`option_scores` is empty or not a dictionary. Cannot select an answer.")

        with patch.object(logging, 'warning') as mock_log_warning:
            self.assertIsNone(select_answer([('A', 0.5)])) # Test with a list of tuples
            mock_log_warning.assert_called_with("`option_scores` is empty or not a dictionary. Cannot select an answer.")


if __name__ == '__main__':
    unittest.main()
