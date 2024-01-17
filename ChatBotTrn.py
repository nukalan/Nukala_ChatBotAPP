# ChatBotTrn.py
import os
from ChatBotTrnTst import ChatBot  # Import the ChatBot class from ChatBotTrnTst module
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    # Define paths
    training_data_path = "data/training_data/"
    model_save_path = "data/model/chatbot_model.bin"
    vector_db_path = "vector_index.annoy"

    # Ensure the directory for saving the model exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    # Load the GPT-2 model and tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Create an instance of the ChatBot class
    chatbot = ChatBot(model_save_path, vector_db_path)
    
    # Load the vector database for training
    chatbot.load_vector_db()

    # Train the chatbot
    chatbot.train_chatbot(training_data_path, model_save_path)
    chatbot.save_vector_db()  # Save the vector database after training

if __name__ == "__main__":
    main()
