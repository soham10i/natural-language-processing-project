# tests/common.py
# This file can be used for shared mock data, helper functions,
# or common setup for tests.
# For now, it can remain empty or include basic comments.

# Example:
# MOCK_DEVICE = "cpu"
# def get_mock_sentence_transformer_model():
#     # In a real scenario, you might mock the SentenceTransformer
#     # to avoid actual model loading during unit tests.
#     class MockSentenceTransformer:
#         def encode(self, sentences, device, convert_to_tensor):
#             # Return dummy embeddings
#             import torch
#             if isinstance(sentences, str):
#                 return torch.rand((384,)) if convert_to_tensor else np.random.rand(384)
#             else:
#                 return torch.rand((len(sentences), 384)) if convert_to_tensor else np.random.rand(len(sentences), 384)
#         def to(self, device):
#             return self
#     return MockSentenceTransformer()

# def get_mock_spacy_model():
#     # Mock Spacy model
#     class MockSpacyDoc:
#         def __init__(self, text, ents):
#             self.text = text
#             self.ents = [MockSpacyEnt(ent_text, ent_label) for ent_text, ent_label in ents]
#     class MockSpacyEnt:
#         def __init__(self, text, label_):
#             self.text = text
#             self.label_ = label_
#     class MockSpacyModel:
#         def __call__(self, text):
#             # Simple mock: return no entities, or predefined ones based on text
#             if "diabetes" in text.lower():
#                 return MockSpacyDoc(text, [("diabetes", "DISEASE")])
#             return MockSpacyDoc(text, [])
#     return MockSpacyModel()

pass
