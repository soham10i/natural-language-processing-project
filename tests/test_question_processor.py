import unittest
from unittest.mock import MagicMock, patch
import torch
# import spacy # Not strictly needed as we mock the spacy model object itself
from src.question_processor import process_question

class TestQuestionProcessor(unittest.TestCase):

    def setUp(self):
        """Set up common test data and mocks."""
        self.sample_question_data_full = {
            "id": "q1",
            "question": "What is Type 1 Diabetes mellitus?",
            "options": {"A": "Condition Alpha", "B": "Condition Beta", "C": "A virus"},
            "metamap_phrases": ["Type 1 Diabetes mellitus", "Diabetes"], # "Diabetes" is a subset, "Type 1 Diabetes mellitus" is exact
            "answer_idx": "A"
        }
        self.device = torch.device("cpu")

        # Mock for nlp_spacy (SciSpacy model)
        self.mock_nlp_spacy = MagicMock()

        # Mock for sentence_model (SentenceTransformer)
        self.mock_sentence_model = MagicMock()
        self.dummy_embedding = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        self.mock_sentence_model.encode.return_value = self.dummy_embedding
        
        # Define side effects for nlp_spacy mock
        # This function will be called when mock_nlp_spacy("some text") is called
        def nlp_side_effect(text_input):
            doc = MagicMock()
            doc.text = text_input # Store text for debugging if needed
            if "What is Type 1 Diabetes mellitus?" in text_input:
                ent1 = MagicMock()
                ent1.text = "Type 1 Diabetes mellitus" # NER finds the full phrase
                doc.ents = [ent1]
            elif "Condition Alpha" in text_input:
                ent_optA = MagicMock()
                ent_optA.text = "Condition Alpha"
                doc.ents = [ent_optA]
            elif "Condition Beta" in text_input:
                ent_optB = MagicMock()
                ent_optB.text = "Condition Beta" # NER might find this
                doc.ents = [ent_optB]
            elif "A virus" in text_input: # Option C, assume NER finds "virus"
                ent_optC = MagicMock()
                ent_optC.text = "virus"
                doc.ents = [ent_optC]
            else:
                doc.ents = [] # Default to no entities for other texts
            return doc

        self.mock_nlp_spacy.side_effect = nlp_side_effect


    def test_successful_processing_with_mocks(self):
        """Test normal processing with metamap phrases and NER entities."""
        result = process_question(self.sample_question_data_full, self.mock_nlp_spacy, self.mock_sentence_model, self.device)

        # Assertions for nlp_spacy calls
        self.mock_nlp_spacy.assert_any_call(self.sample_question_data_full['question'])
        self.mock_nlp_spacy.assert_any_call(self.sample_question_data_full['options']['A'])
        self.mock_nlp_spacy.assert_any_call(self.sample_question_data_full['options']['B'])
        self.mock_nlp_spacy.assert_any_call(self.sample_question_data_full['options']['C'])
        self.assertEqual(self.mock_nlp_spacy.call_count, 1 + len(self.sample_question_data_full['options']))

        # Assertion for sentence_model.encode call
        cleaned_question_expected = self.sample_question_data_full['question'].lower()
        self.mock_sentence_model.encode.assert_called_once_with(
            cleaned_question_expected,
            device=self.device,
            convert_to_tensor=True
        )

        # Assertions for the returned dictionary structure
        self.assertIn('original_question_data', result)
        self.assertIn('extracted_entities', result)
        self.assertIn('question_embedding', result)

        # Verify original_question_data
        self.assertEqual(result['original_question_data'], self.sample_question_data_full)

        # Verify extracted_entities (order doesn't matter, so use sets for comparison)
        expected_entities = {"Type 1 Diabetes mellitus", "Diabetes", "Condition Alpha", "Condition Beta", "virus"}
        self.assertSetEqual(set(result['extracted_entities']), expected_entities)
        
        # Verify question_embedding
        self.assertTrue(torch.equal(result['question_embedding'], self.dummy_embedding))

    def test_handling_no_metamap_phrases(self):
        """Test processing when 'metamap_phrases' is missing or empty."""
        question_data_no_metamap = self.sample_question_data_full.copy()
        del question_data_no_metamap['metamap_phrases'] # Test missing key

        result = process_question(question_data_no_metamap, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
        expected_entities_from_ner = {"Type 1 Diabetes mellitus", "Condition Alpha", "Condition Beta", "virus"}
        self.assertSetEqual(set(result['extracted_entities']), expected_entities_from_ner)

        question_data_empty_metamap = self.sample_question_data_full.copy()
        question_data_empty_metamap['metamap_phrases'] = [] # Test empty list
        
        # Reset call counts for mocks if necessary, or use new mocks. For simplicity, reusing.
        self.mock_nlp_spacy.reset_mock() 
        self.mock_sentence_model.encode.reset_mock()
        
        result_empty = process_question(question_data_empty_metamap, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
        self.assertSetEqual(set(result_empty['extracted_entities']), expected_entities_from_ner)


    def test_handling_no_spacy_entities(self):
        """Test processing when Spacy NER returns no entities."""
        mock_nlp_no_entities = MagicMock()
        mock_nlp_no_entities.side_effect = lambda text: MagicMock(ents=[]) # Always returns no entities

        result = process_question(self.sample_question_data_full, mock_nlp_no_entities, self.mock_sentence_model, self.device)
        
        # Expected entities should only be from metamap_phrases
        expected_entities_from_metamap = set(self.sample_question_data_full['metamap_phrases'])
        self.assertSetEqual(set(result['extracted_entities']), expected_entities_from_metamap)
        # Ensure spacy mock was called for question and options
        self.assertEqual(mock_nlp_no_entities.call_count, 1 + len(self.sample_question_data_full['options']))


    def test_question_cleaning_lowercase(self):
        """Test that the question is lowercased before encoding."""
        mixed_case_question_data = {
            "question": "WHAT is Type 1 DIABETES Mellitus?", # Mixed case
            "options": {"A": "Condition Alpha"},
            "metamap_phrases": ["Type 1 Diabetes mellitus"]
        }
        
        # Reset sentence_model mock to check its call argument
        self.mock_sentence_model.encode.reset_mock()
        # Need a spacy mock that handles the new question text
        mock_nlp_for_mixed_case = MagicMock()
        def nlp_side_effect_mixed(text_input):
            doc = MagicMock()
            if "WHAT is Type 1 DIABETES Mellitus?" in text_input:
                doc.ents = [MagicMock(text="Type 1 DIABETES Mellitus")]
            elif "Condition Alpha" in text_input:
                doc.ents = [MagicMock(text="Condition Alpha")]
            else:
                doc.ents = []
            return doc
        mock_nlp_for_mixed_case.side_effect = nlp_side_effect_mixed

        process_question(mixed_case_question_data, mock_nlp_for_mixed_case, self.mock_sentence_model, self.device)

        expected_cleaned_question = "what is type 1 diabetes mellitus?"
        self.mock_sentence_model.encode.assert_called_once_with(
            expected_cleaned_question,
            device=self.device,
            convert_to_tensor=True
        )

    def test_missing_question_or_options_keys(self):
        """Test behavior when 'question' or 'options' keys are missing from input."""
        # Test missing 'question'
        data_no_question = {"options": {"A": "Test"}, "metamap_phrases": []}
        with patch('src.question_processor.logging.error') as mock_log_error:
            result = process_question(data_no_question, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
            self.assertIsNone(result) # Function should return None
            mock_log_error.assert_called_with("Missing 'question' or 'options' in question_data.")

        # Test missing 'options'
        data_no_options = {"question": "A test question?", "metamap_phrases": []}
        with patch('src.question_processor.logging.error') as mock_log_error:
            result = process_question(data_no_options, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
            self.assertIsNone(result) # Function should return None
            mock_log_error.assert_called_with("Missing 'question' or 'options' in question_data.")

    def test_options_not_a_dict(self):
        """Test when 'options' is not a dictionary (e.g. a string or list)."""
        question_data_invalid_options = {
            "question": "What is Type 1 Diabetes mellitus?",
            "options": "This is not a dict", # Invalid options format
            "metamap_phrases": ["Type 1 Diabetes mellitus", "Diabetes"]
        }
        with patch('src.question_processor.logging.warning') as mock_log_warning:
            result = process_question(question_data_invalid_options, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
            # NER on options should be skipped, but question processing should proceed
            self.assertIsNotNone(result)
            self.assertIn('extracted_entities', result)
            expected_entities = {"Type 1 Diabetes mellitus", "Diabetes"} # Only from question NER + metamap
            self.assertSetEqual(set(result['extracted_entities']), expected_entities)
            mock_log_warning.assert_any_call(f"Options for question '{question_data_invalid_options['question'][:50]}...' are not a dictionary, skipping NER on options.")

    def test_option_value_not_a_string(self):
        """Test when an option's value is not a string."""
        question_data_invalid_option_value = {
            "question": "What is Type 1 Diabetes mellitus?",
            "options": {"A": "Valid Option", "B": 12345}, # Option B is not a string
            "metamap_phrases": ["Type 1 Diabetes mellitus", "Diabetes"]
        }
        
        # Reset and reconfigure nlp_spacy mock for this specific test's inputs
        self.mock_nlp_spacy.reset_mock()
        def nlp_side_effect_specific(text_input):
            doc = MagicMock()
            if "What is Type 1 Diabetes mellitus?" in text_input:
                doc.ents = [MagicMock(text="Type 1 Diabetes mellitus")]
            elif "Valid Option" in text_input:
                 doc.ents = [MagicMock(text="Valid Option")]
            # No rule for 12345 as it won't be passed to spacy if it's not a string
            else:
                doc.ents = []
            return doc
        self.mock_nlp_spacy.side_effect = nlp_side_effect_specific

        with patch('src.question_processor.logging.warning') as mock_log_warning:
            result = process_question(question_data_invalid_option_value, self.mock_nlp_spacy, self.mock_sentence_model, self.device)
            self.assertIsNotNone(result)
            expected_entities = {"Type 1 Diabetes mellitus", "Diabetes", "Valid Option"}
            self.assertSetEqual(set(result['extracted_entities']), expected_entities)
            mock_log_warning.assert_any_call(f"Option 'B' for question '{question_data_invalid_option_value['question'][:50]}...' is not a string, skipping NER.")


if __name__ == '__main__':
    unittest.main()
