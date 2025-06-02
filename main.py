import torch
import argparse
import logging
import spacy
from sentence_transformers import SentenceTransformer

# Import custom modules
from src.data_loader import load_medqa_data
from src.question_processor import process_question
from src.wikipedia_selector import select_wikipedia_pages
from src.evidence_scorer import score_options_similarity_with_vector_db
from src.answer_selector import select_answer
# Optional: Placeholder imports for future fine-tuning steps
# from src.distilbert_classifier import fine_tune_distilbert_classifier
# from src.distilbert_nli import fine_tune_distilbert_nli

# --- Logging Setup ---
# BasicConfig should be called only once, preferably at the very start of your application.
# The format now includes %(name)s for the logger name.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # --- Device Selection (already in the original template but good to confirm) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Command-line Argument Parsing ---
    parser = argparse.ArgumentParser(description="MedQA Question Answering Pipeline.")
    parser.add_argument("dataset_file_path", help="Path to the MedQA dataset file (JSONL format).")
    parser.add_argument("--subset_1000", action="store_true", help="Use only a subset of 1000 samples from the dataset.")
    parser.add_argument("--max_wiki_pages", type=int, default=5, help="Maximum number of Wikipedia pages to retrieve per question.")
    parser.add_argument("--faiss_top_k", type=int, default=1, help="Top K evidence sentences to consider from FAISS search for scoring.")
    # Add arguments for model paths if they were being loaded instead of downloaded, or for fine-tuning triggers
    # parser.add_argument("--scispaCy_model_name", type=str, default="en_core_sci_sm", help="Name of the SciSpacy model to load.")
    # parser.add_argument("--sentence_transformer_model_name", type=str, default="all-MiniLM-L6-v2", help="Name of the Sentence Transformer model.")

    args = parser.parse_args()
    logger.info(f"Dataset file path: {args.dataset_file_path}")
    logger.info(f"Subset 1000: {args.subset_1000}")
    logger.info(f"Max Wikipedia pages per question: {args.max_wiki_pages}")
    logger.info(f"FAISS top K for evidence: {args.faiss_top_k}")

    # --- Model Initialization ---
    try:
        logger.info("Loading SciSpacy model 'en_core_sci_sm'...")
        nlp_spacy = spacy.load("en_core_sci_sm")
        logger.info("SciSpacy model loaded successfully.")
    except OSError as e:
        logger.critical(f"Failed to load SciSpacy model 'en_core_sci_sm'. Error: {e}", exc_info=True)
        logger.critical("Please ensure 'en_core_sci_sm' is installed (e.g., python -m spacy download en_core_sci_sm). Exiting.")
        return # Exit if essential models fail to load

    try:
        logger.info("Loading Sentence Transformer model 'all-MiniLM-L6-v2'...")
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_model.to(device) # Move model to the selected device
        logger.info(f"Sentence Transformer model loaded successfully and moved to {device}.")
    except Exception as e:
        logger.critical(f"Failed to load or move Sentence Transformer model. Error: {e}", exc_info=True)
        logger.critical("Exiting application due to model loading failure.")
        return # Exit

    # --- Data Loading ---
    logger.info(f"Loading MedQA data from: {args.dataset_file_path}")
    medqa_questions = load_medqa_data(args.dataset_file_path)

    if not medqa_questions: # load_medqa_data returns [] on error or no data
        logger.error("No questions loaded from dataset file. Please check the file path and format. Exiting.")
        return

    num_total_questions = len(medqa_questions)
    if args.subset_1000:
        medqa_questions = medqa_questions[:1000]
        logger.info(f"Using a subset of {len(medqa_questions)} questions out of {num_total_questions} total.")
    else:
        logger.info(f"Loaded {num_total_questions} questions in total.")

    # --- Main Processing Loop ---
    correct_predictions = 0
    questions_actually_processed_for_accuracy = 0 # Questions for which an answer was attempted
    questions_skipped_due_to_missing_answer_idx = 0

    for i, question_data in enumerate(medqa_questions):
        question_id = question_data.get('id', f'index_{i}')
        logger.info(f"--- Processing question {i+1}/{len(medqa_questions)} (ID: {question_id}) ---")
        
        actual_answer_key = question_data.get('answer_idx')
        if actual_answer_key is None:
            logger.warning(f"Question ID {question_id} is missing 'answer_idx'. Skipping evaluation for this question.")
            questions_skipped_due_to_missing_answer_idx +=1
            continue # Skip to next question if we can't evaluate it

        questions_actually_processed_for_accuracy += 1
        predicted_answer_key = None # Initialize predicted answer for this iteration

        try:
            # 1. Process Question (NER, Embeddings)
            logger.debug(f"Question ID {question_id}: Processing with question_processor...")
            processed_q = process_question(question_data, nlp_spacy, sentence_model, device)
            if not processed_q or 'question_embedding' not in processed_q or 'extracted_entities' not in processed_q:
                logger.warning(f"Question ID {question_id}: Failed to process (e.g., missing keys from process_question). Skipping further steps for this question.")
                continue

            # 2. Select Wikipedia Pages
            logger.debug(f"Question ID {question_id}: Selecting Wikipedia pages based on entities: {processed_q['extracted_entities']}...")
            wiki_pages_texts = select_wikipedia_pages(processed_q, sentence_model, device, max_pages=args.max_wiki_pages)
            if not wiki_pages_texts:
                logger.warning(f"Question ID {question_id}: No Wikipedia pages retrieved. Cannot score options. This will count as an incorrect prediction.")
                continue

            # 3. Score Options with Vector DB (FAISS)
            logger.debug(f"Question ID {question_id}: Scoring options against {len(wiki_pages_texts)} Wikipedia page(s)...")
            option_scores = score_options_similarity_with_vector_db(processed_q, wiki_pages_texts, sentence_model, device, top_k_evidence=args.faiss_top_k)
            if not option_scores:
                logger.warning(f"Question ID {question_id}: Option scoring returned empty. Cannot select an answer. This will count as an incorrect prediction.")
                continue
            logger.info(f"Question ID {question_id}: Option scores: {option_scores}")

            # 4. Select Answer
            logger.debug(f"Question ID {question_id}: Selecting best answer based on scores...")
            predicted_answer_key = select_answer(option_scores) # This is the variable we check later
            if predicted_answer_key is None:
                logger.warning(f"Question ID {question_id}: Could not select an answer from scores (select_answer returned None). This will count as an incorrect prediction.")
                # No prediction made, so it's incorrect by default for accuracy calculation
                continue 
            
            logger.info(f"Question ID {question_id}: Predicted Answer Key: '{predicted_answer_key}', Correct Answer Key: '{actual_answer_key}'")

            # 5. Evaluate Prediction
            if predicted_answer_key == actual_answer_key:
                correct_predictions += 1
                logger.info(f"Question ID {question_id}: Prediction CORRECT.")
            else:
                logger.info(f"Question ID {question_id}: Prediction INCORRECT.")

        except Exception: # Catch any unexpected error during a single question's processing
            logger.exception(f"Question ID {question_id}: An unexpected error occurred during processing. This will count as an incorrect prediction.")
            # 'continue' is implicit if this is the last part of the loop, error is logged, question is marked incorrect
        finally:
            logger.info(f"--- Finished processing question {i+1}/{len(medqa_questions)} (ID: {question_id}) ---")


    # --- Evaluation ---
    logger.info("================== Overall Results ==================")
    logger.info(f"Total questions in dataset (or subset): {len(medqa_questions)}")
    logger.info(f"Questions skipped due to missing 'answer_idx': {questions_skipped_due_to_missing_answer_idx}")
    
    if questions_actually_processed_for_accuracy > 0:
        accuracy = correct_predictions / questions_actually_processed_for_accuracy
        logger.info(f"Total questions evaluated for accuracy: {questions_actually_processed_for_accuracy}")
        logger.info(f"Correct predictions: {correct_predictions}")
        logger.info(f"Overall Accuracy: {accuracy:.4f} ({correct_predictions}/{questions_actually_processed_for_accuracy})")
        print(f"\nOverall Accuracy: {accuracy:.4f} ({correct_predictions}/{questions_actually_processed_for_accuracy})")
    else:
        logger.warning("No questions were processed and evaluated for accuracy (e.g., all questions might have been skipped or the dataset was empty).")
        print("No questions were processed and evaluated for accuracy.")
    logger.info("===================================================")


if __name__ == "__main__":
    # --- Potentially trigger fine-tuning scripts here if needed ---
    # For example, based on some arguments or conditions:
    # if args.run_classifier_training:
    #     logging.info("Starting DistilBERT classifier fine-tuning (placeholder)...")
    #     fine_tune_distilbert_classifier("path/to/classifier_traindata.csv", "./models/distilbert_classifier_finetuned", device)
    # if args.run_nli_training:
    #     logging.info("Starting DistilBERT NLI fine-tuning (placeholder)...")
    #     fine_tune_distilbert_nli("path/to/nli_traindata.jsonl", "./models/distilbert_nli_finetuned", device)
    
    main()
