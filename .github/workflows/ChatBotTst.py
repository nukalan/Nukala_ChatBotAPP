# ChatBotTst.py

from ChatBotTrnTst import ChatBot  # Import the ChatBot class from ChatBotTrnTst module

def test_chatbot(chatbot, user_queries):
    for query in user_queries:
        response = chatbot.generate_response(query)
        print(f"User: {query}")
        print(f"ChatBot: {response}\n")

def main():
    # Define paths
    model_directory = "data/model/chatbot_model/"  # Change to the folder containing the model
    vector_db_path = "vector_index.annoy"

    # Create an instance of the ChatBot class
    chatbot = ChatBot(model_directory, vector_db_path)

    # Load the vector database for testing
    chatbot.load_vector_db()

    # Generate synthetic queries and test the chatbot
    synthetic_queries = ["What is the meaning of life?", "Tell me a joke.", "Explain quantum physics."]
    test_chatbot(chatbot, synthetic_queries)

if __name__ == "__main__":
    main()
