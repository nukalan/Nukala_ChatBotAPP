# ChatBotRtrnTst.py

from ChatBotTrnTst import ChatBot  # Import the ChatBot class from ChatBotTrnTst module
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from annoy_vector_database import AnnoyVectorDatabase

def test_chatbot(chatbot, test_queries):
    for query in test_queries:
        response = chatbot.generate_response(query)
        print(f"User: {query}")
        print(f"ChatBot: {response}\n")

def retrain_chatbot(chatbot, additional_training_data, model_save_path):
    # Load additional training data (if any)
    # You might want to customize this based on your actual data loading mechanism
    # Additional training data should be in a format suitable for your training logic
    # For example, it could be a list of dialogue pairs.

    # Update the train_chatbot method in ChatBotTrnTst.py to handle additional data
    chatbot.train_chatbot(additional_training_data, model_save_path)

def main():
    # Initialize the chatbot
    model_path = "data/model/chatbot_model.bin"
    vector_db_path = "vector_index.annoy"

    chatbot = ChatBot(model_path, vector_db_path)

    # Test the chatbot with synthetic queries
    synthetic_queries = ["What is the meaning of life?", "Tell me a joke.", "Explain quantum physics."]
    test_chatbot(chatbot, synthetic_queries)

    # Retrain the chatbot with additional training data (if available)
    additional_training_data = ["New training data 1", "New training data 2"]
    model_save_path = "data/model/chatbot_model_updated.bin"
    retrain_chatbot(chatbot, additional_training_data, model_save_path)

    # Test the updated chatbot
    test_queries_after_retrain = ["After retraining: What is the meaning of life?", "After retraining: Tell me a joke."]
    test_chatbot(chatbot, test_queries_after_retrain)

if __name__ == "__main__":
    main()

