# rag_framework.py
from gpt2_language_model import GPT2LanguageModel
from annoy_vector_database import AnnoyVectorDatabase


def retrieve_information_using_vector_db(input_texts, vector_db, num_neighbors=5, allow_self_query=True):
    query_vectors = vector_db.get_vectors_from_texts(input_texts)
    retrieved_indices = vector_db.query_vectors(query_vectors, k=num_neighbors, include_self=allow_self_query)
    print("Query Vectors:", query_vectors)
    print("Retrieved Indices:", retrieved_indices)
    return retrieved_indices

def generate_response_using_language_model(input_text, language_model, max_length=50, num_beams=5, temperature=0.7):
    """
    Generate a response using the language model.

    Args:
        input_text (str): The input text.
        language_model (GPT2LanguageModel): The language model.
        max_length (int): Maximum length of the generated text.
        num_beams (int): Number of beams for beam search.
        temperature (float): Sampling temperature.

    Returns:
        str: The generated text.
    """
    return language_model.generate_text(input_text, max_length=max_length, num_beams=num_beams, temperature=temperature)

def generate_rag_output(input_texts, language_model, vector_db, num_neighbors=5, allow_self_query=True):
    query_vectors = vector_db.get_vectors_from_texts(input_texts)
    # Generate text using language model
    generated_text = generate_response_using_language_model(input_texts[0], language_model)
    print("LLM Output:", generated_text)

    # Retrieve relevant information using vector database
    retrieved_indices = retrieve_information_using_vector_db(input_texts, vector_db, num_neighbors, allow_self_query)
    
    # Add a print statement to check the value of is_from_pdf
    print("Is from PDF:", vector_db.index_loaded)

    # Convert retrieved indices to text format
    retrieved_indices_text = [f"Index {index}: {vector_db.get_text_from_index(index)}" for index in retrieved_indices]
    print("VB Output:", retrieved_indices_text)

    # Combine generated text and retrieved information into a dictionary
    final_output = {
        "generated_text": generated_text,
        "retrieved_indices": retrieved_indices_text,
        "is_from_pdf": vector_db.index_loaded  # Modify this based on your logic
    }

    return final_output
