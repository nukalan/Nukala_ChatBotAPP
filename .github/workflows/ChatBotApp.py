# ChatBotApp.py

import numpy as np
from flask import Flask, render_template, request, jsonify
from rag_framework import generate_rag_output, GPT2LanguageModel, AnnoyVectorDatabase
from pdf_loader import extract_text_from_pdf
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from annoy_vector_database import AnnoyVectorDatabase
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = 'uploads'  # create a folder named 'uploads' in your project directory
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained GPT-2 model
language_model = GPT2LanguageModel()

# Load pre-trained vectors or train your custom model
vectors = np.random.rand(100, 50).astype('float32')  # Example vectors
vector_db = AnnoyVectorDatabase(vectors)

###NUKALA
import json

# Function to dump the text content of the vector database
def dump_vector_database(vector_db):
    data_to_dump = {}

    index = 0
    while True:
        try:
            text = vector_db.get_text(index)
            data_to_dump[text] = index
            index += 1
        except Exception as e:
            # Break the loop when an exception occurs (indicating the end of the database)
            break

    with open('vector_database_dump.json', 'w') as file:
        json.dump(data_to_dump, file)

# Load proprietary PDF documents during initialization
proprietary_pdf_files = ["uploads/MindMastery.pdf", "uploads/MindChange.pdf","uploads/MindHacking.pdf"]

try:
    # Check if Annoy index is loaded
    if not vector_db.index_loaded:
        print("Loading PDF files into the vector database...")
        for pdf_file_path in proprietary_pdf_files:
            pdf_text = extract_text_from_pdf(pdf_file_path)
            vector_db.upload_pdf_to_vector_db(pdf_text)
        print("PDF files loaded into the vector database.")
except Exception as e:
    print(f"An error occurred: {e}")

@app.route('/')
def home():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def remove_repetitive_responses(rag_output):
    # Check if rag_output is a dictionary and has the expected keys
    if isinstance(rag_output, dict) and all(key in rag_output for key in ["generated_text", "retrieved_indices", "is_from_pdf"]):
        # Split the generated text into sentences
        sentences = rag_output["generated_text"].split('. ')

        # Create a list to store unique sentences
        unique_sentences = []

        # Iterate through sentences and add only unique ones
        for sentence in sentences:
            if sentence not in unique_sentences:
                unique_sentences.append(sentence)

        # Join the unique sentences back into text
        filtered_text = '. '.join(unique_sentences)

        return filtered_text
    else:
        # Handle the case where rag_output is not in the expected format
        return "Error: Unexpected rag_output format"

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']

    # Upload PDF to Vector Database
    if 'pdf_file' in request.files:
        pdf_file = request.files['pdf_file']
        if pdf_file and allowed_file(pdf_file.filename):
            filename = secure_filename(pdf_file.filename)
            pdf_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pdf_text = extract_text_from_pdf(pdf_path)

            # Upload PDF to Vector Database with a tag
            vector_db.upload_pdf_to_vector_db(pdf_text, tag="pdf")

   # Multivector and multiple queries supported
    rag_output = generate_rag_output([user_input], language_model, vector_db, num_neighbors=5)
    print("RAG Output:", rag_output)
    filtered_rag_output = remove_repetitive_responses(rag_output)
    print("RAG Output:", filtered_rag_output)

    # Check if the response is from PDF
    is_from_pdf = rag_output.get("is_from_pdf", False)
    print("Response is from PDF:", is_from_pdf)

###NUKALA
    # Dump vector database content for debugging
    dump_vector_database(vector_db)

    return jsonify({"user_input": user_input, "rag_output": filtered_rag_output, "is_from_pdf": is_from_pdf})

if __name__ == '__main__':
    app.run(debug=True)
