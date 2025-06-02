import unittest
from unittest.mock import patch, MagicMock, call
import torch
import numpy as np
# import faiss # Not strictly needed as we mock the faiss module object itself
# import nltk # Not strictly needed as we mock functions from it

# Import the function to be tested
from src.evidence_scorer import score_options_similarity_with_vector_db

# Mock NLTK download if it's directly in evidence_scorer.py's global scope (it is)
# This needs to be done before 'from src.evidence_scorer import ...' if nltk.download is at module level.
# However, since it's inside a try-except block, we can patch it where it's used or at class/method level.

class TestEvidenceScorer(unittest.TestCase):

    def setUp(self):
        """Set up common test data and mocks."""
        self.device_cpu = torch.device("cpu")
        self.device_cuda = torch.device("cuda") # For testing GPU path

        self.sample_processed_question = {
            'original_question_data': {
                'question': "What is diabetes type 2?",
                'options': {'A': "A metabolic disorder", 'B': "An infectious disease"}
            },
            # Dummy embedding, actual dimension from sentence_model is 384 for all-MiniLM-L6-v2
            # For test simplicity, we'll use a smaller dimension, e.g., 3, for our dummy embeddings.
            'question_embedding': torch.rand(1, 3) # Shape (1, embedding_dim)
        }
        self.embedding_dim = 3 # Must match the dummy embeddings

        self.wikipedia_texts = ["Diabetes mellitus type 2 is a long-term metabolic disorder.",
                                "It is characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
                                "Common symptoms include increased thirst, frequent urination, and unexplained weight loss."]

        # Mock SentenceTransformer model
        self.mock_sentence_model = MagicMock()

        # Define a more flexible side_effect for sentence_model.encode
        def encode_side_effect(input_texts, device, convert_to_tensor, batch_size=32, show_progress_bar=False):
            if isinstance(input_texts, str): # Single query string
                # For query "What is diabetes type 2? <SEP> A metabolic disorder"
                # or "What is diabetes type 2? <SEP> An infectious disease"
                raw_embedding = np.random.rand(self.embedding_dim).astype(np.float32)
            elif isinstance(input_texts, list): # List of segments
                raw_embedding = np.random.rand(len(input_texts), self.embedding_dim).astype(np.float32)
            else:
                raise TypeError(f"Mock encode received unexpected type: {type(input_texts)}")
            
            # Normalize for IndexFlatIP
            if raw_embedding.ndim == 1:
                norm = np.linalg.norm(raw_embedding)
                if norm == 0: norm = 1e-12 # Avoid division by zero
                normalized_embedding = raw_embedding / norm
            else: # 2D
                norms = np.linalg.norm(raw_embedding, axis=1, keepdims=True)
                norms[norms == 0] = 1e-12 # Avoid division by zero
                normalized_embedding = raw_embedding / norms

            return torch.from_numpy(normalized_embedding).to(device) if convert_to_tensor else normalized_embedding

        self.mock_sentence_model.encode.side_effect = encode_side_effect
        
        # Mock for faiss index instance, to be redefined in tests needing specific faiss behavior
        self.mock_faiss_index_instance = MagicMock()


    # Patch nltk.sent_tokenize and faiss for all tests in this class
    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize') # sent_tokenize is imported directly
    @patch('src.evidence_scorer.nltk.download') # To prevent actual download attempts
    def test_successful_scoring_cpu_faiss(self, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test successful scoring path using CPU FAISS."""
        mock_sent_tokenize.return_value = [
            "Diabetes mellitus type 2 is a long-term metabolic disorder.",
            "It is characterized by high blood sugar, insulin resistance, and relative lack of insulin.",
            "Common symptoms include increased thirst, frequent urination, and unexplained weight loss."
        ]
        
        # Configure mock_faiss_module behavior
        mock_faiss_module.IndexFlatIP.return_value = self.mock_faiss_index_instance
        
        # Simulate FAISS search results: (Distances, Indices)
        # Assuming top_k_evidence=1. Score should be D[0][0]
        # For two options, search will be called twice.
        self.mock_faiss_index_instance.search.side_effect = [
            (np.array([[0.95]], dtype=np.float32), np.array([[0]])), # For option A
            (np.array([[0.88]], dtype=np.float32), np.array([[1]]))  # For option B
        ]
        self.mock_faiss_index_instance.ntotal = len(mock_sent_tokenize.return_value)


        option_scores = score_options_similarity_with_vector_db(
            self.sample_processed_question,
            self.wikipedia_texts,
            self.mock_sentence_model,
            self.device_cpu
        )

        # Assertions
        mock_nltk_download.assert_called_once_with('punkt', quiet=True) # Check if punkt download was managed
        mock_sent_tokenize.assert_called_once_with("\n\n".join(self.wikipedia_texts))
        
        # Check sentence_model.encode calls
        # 1 call for segments, 2 calls for the two options
        self.assertEqual(self.mock_sentence_model.encode.call_count, 1 + len(self.sample_processed_question['original_question_data']['options']))
        
        # FAISS assertions
        mock_faiss_module.IndexFlatIP.assert_called_once_with(self.embedding_dim)
        self.mock_faiss_index_instance.add.assert_called_once() # Check that segment embeddings were added
        # Check that segment embeddings passed to add are numpy and normalized
        added_embeddings = self.mock_faiss_index_instance.add.call_args[0][0]
        self.assertIsInstance(added_embeddings, np.ndarray)
        # Check normalization (norms should be close to 1)
        np.testing.assert_allclose(np.linalg.norm(added_embeddings, axis=1), 1.0, rtol=1e-5)


        self.assertEqual(self.mock_faiss_index_instance.search.call_count, 2) # Called for each option

        # Verify scores
        self.assertIn('A', option_scores)
        self.assertIn('B', option_scores)
        self.assertAlmostEqual(option_scores['A'], 0.95, places=5)
        self.assertAlmostEqual(option_scores['B'], 0.88, places=5)

        # Ensure GPU resources were not called
        mock_faiss_module.StandardGpuResources.assert_not_called()
        mock_faiss_module.index_cpu_to_gpu.assert_not_called()


    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize')
    @patch('src.evidence_scorer.nltk.download')
    def test_gpu_faiss_path(self, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test successful scoring path using GPU FAISS (if faiss-gpu were available)."""
        mock_sent_tokenize.return_value = ["Sentence 1 for GPU.", "Sentence 2 for GPU."]
        
        mock_gpu_index_instance = MagicMock()
        mock_faiss_module.IndexFlatIP.return_value = self.mock_faiss_index_instance # CPU index first
        mock_faiss_module.StandardGpuResources.return_value = MagicMock() # Mock GPU resources
        mock_faiss_module.index_cpu_to_gpu.return_value = mock_gpu_index_instance # This is the index used on GPU

        mock_gpu_index_instance.search.side_effect = [
            (np.array([[0.92]], dtype=np.float32), np.array([[0]])),
            (np.array([[0.85]], dtype=np.float32), np.array([[1]]))
        ]
        mock_gpu_index_instance.ntotal = len(mock_sent_tokenize.return_value)


        option_scores = score_options_similarity_with_vector_db(
            self.sample_processed_question,
            self.wikipedia_texts, # Needs some text
            self.mock_sentence_model,
            self.device_cuda # Specify CUDA device
        )

        mock_faiss_module.StandardGpuResources.assert_called_once()
        mock_faiss_module.index_cpu_to_gpu.assert_called_once()
        # Ensure .add and .search were called on the GPU index instance
        mock_gpu_index_instance.add.assert_called_once()
        self.assertEqual(mock_gpu_index_instance.search.call_count, 2)
        
        self.assertAlmostEqual(option_scores['A'], 0.92, places=5)


    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize')
    @patch('src.evidence_scorer.nltk.download')
    @patch('src.evidence_scorer.logging') # Mock logging module
    def test_no_wikipedia_texts(self, mock_logging, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test behavior when wikipedia_texts list is empty."""
        option_scores = score_options_similarity_with_vector_db(
            self.sample_processed_question,
            [], # Empty list of wiki texts
            self.mock_sentence_model,
            self.device_cpu
        )
        
        expected_scores = {key: 0.0 for key in self.sample_processed_question['original_question_data']['options']}
        self.assertEqual(option_scores, expected_scores)
        mock_logging.warning.assert_any_call("No Wikipedia texts provided for scoring.")
        mock_faiss_module.IndexFlatIP.assert_not_called() # FAISS index should not be created


    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize')
    @patch('src.evidence_scorer.nltk.download')
    @patch('src.evidence_scorer.logging')
    def test_no_segments_after_tokenization(self, mock_logging, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test behavior when sent_tokenize returns no usable segments."""
        mock_sent_tokenize.return_value = ["short", "tiny"] # These will be filtered out
        
        # Mock the fallback splitting by paragraph to also return no usable segments
        with patch('src.evidence_scorer.re.split', return_value=[]): # If re.split was used for fallback
             # The current implementation uses full_text.split('\n\n')
             # To mock this, we need to ensure wikipedia_texts makes full_text.split yield no segments
             short_wiki_texts = ["word1\n\nword2"] # These parts are too short

             option_scores = score_options_similarity_with_vector_db(
                self.sample_processed_question,
                short_wiki_texts, 
                self.mock_sentence_model,
                self.device_cpu
            )

        expected_scores = {key: 0.0 for key in self.sample_processed_question['original_question_data']['options']}
        self.assertEqual(option_scores, expected_scores)
        mock_logging.error.assert_any_call("No usable text segments found in Wikipedia texts.")
        mock_faiss_module.IndexFlatIP.assert_not_called()

    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize')
    @patch('src.evidence_scorer.nltk.download')
    def test_faiss_search_returns_no_results(self, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test when FAISS search returns empty results for D."""
        mock_sent_tokenize.return_value = ["A valid sentence for indexing."]
        mock_faiss_module.IndexFlatIP.return_value = self.mock_faiss_index_instance
        self.mock_faiss_index_instance.search.return_value = (np.array([[]]), np.array([[]])) # Empty D
        self.mock_faiss_index_instance.ntotal = 1


        option_scores = score_options_similarity_with_vector_db(
            self.sample_processed_question,
            self.wikipedia_texts,
            self.mock_sentence_model,
            self.device_cpu
        )
        # Expect scores to be 0.0 if no results from FAISS
        self.assertAlmostEqual(option_scores['A'], 0.0, places=5)
        self.assertAlmostEqual(option_scores['B'], 0.0, places=5)

    @patch('src.evidence_scorer.faiss')
    @patch('src.evidence_scorer.sent_tokenize')
    @patch('src.evidence_scorer.nltk.download')
    @patch('src.evidence_scorer.logging')
    def test_option_text_not_string(self, mock_logging, mock_nltk_download, mock_sent_tokenize, mock_faiss_module):
        """Test when an option's text is not a string."""
        mock_sent_tokenize.return_value = ["Some evidence text."]
        mock_faiss_module.IndexFlatIP.return_value = self.mock_faiss_index_instance
        # Simulate search for the valid option
        self.mock_faiss_index_instance.search.return_value = (np.array([[0.77]]), np.array([[0]]))
        self.mock_faiss_index_instance.ntotal = 1


        processed_q_malformed_option = {
            'original_question_data': {
                'question': "What is diabetes type 2?",
                'options': {'A': 12345, 'B': "An infectious disease"} # Option A is not a string
            },
            'question_embedding': torch.rand(1, self.embedding_dim)
        }

        option_scores = score_options_similarity_with_vector_db(
            processed_q_malformed_option,
            self.wikipedia_texts,
            self.mock_sentence_model,
            self.device_cpu
        )

        mock_logging.warning.assert_any_call("Option 'A' text is not a string: 12345. Assigning score 0.")
        self.assertEqual(option_scores['A'], 0.0)
        self.assertAlmostEqual(option_scores['B'], 0.77, places=5) # Valid option should still be scored

if __name__ == '__main__':
    unittest.main()
