# annoy_vector_database.py

from annoy import AnnoyIndex
import numpy as np 
from pdf_loader import extract_text_from_pdf  

import os

class AnnoyVectorDatabase:
    def __init__(self, vectors=None, index_filename="vector_index.annoy", pdf_filename=None):

        if vectors is not None:
            self.dimension = vectors.shape[1]
        else:
            self.dimension = dimension if dimension is not None else 50  # Set a default dimension
        #####self.dimension = vectors.shape[1] if vectors is not None else None
        self.annoy_index = None
        self.index_loaded = False  # Flag to track whether the index is loaded

        if vectors is not None and not self.index_loaded:
            self.annoy_index = AnnoyIndex(self.dimension, 'angular')
            for i in range(len(vectors)):
                self.annoy_index.add_item(i, vectors[i])

            # Build the index
            self.annoy_index.build(10)  # You can adjust the number of trees based on your data

            # Save the index to a file (optional but recommended for reuse)
            self.index_filename = index_filename
            self.annoy_index.save(self.index_filename)
            self.index_loaded = True

            # Print the PDF filename
            if pdf_filename:
                print(f"Annoy index loaded from PDF file: {pdf_filename}")

        elif index_filename and not self.index_loaded:
            self.annoy_index = AnnoyIndex(self.dimension, 'angular')
            self.annoy_index.load(index_filename)
            self.index_loaded = True

            # Print the PDF filename
            if pdf_filename:
                print(f"Annoy index loaded from PDF file-2: {pdf_filename}")

    def query_vectors(self, query_vectors, k=5, include_self=False):
        # Query the index to retrieve similar vectors
        indices = [self.annoy_index.get_nns_by_vector(query_vector, k + 1 if include_self else k, include_distances=False) for query_vector in query_vectors]
        if not include_self:
            indices = [index[1:] for index in indices]

        return indices

    def save_index(self):
        # Save the index to a file (optional but recommended for reuse)
        if not self.index_loaded:
            self.annoy_index.save(self.index_filename)

    def upload_pdf_to_vector_db(self, pdf_text, tag=None):
        vector = self.get_vector_from_text(pdf_text)

        if not self.index_loaded:
            self.annoy_index.add_item(self.annoy_index.get_n_items(), vector)
            self.annoy_index.build(10)  # Rebuild the index after adding a new vector
            print("PDF loaded to vector database.")
        else:
            raise Exception("You can't add an item to a loaded index")

    def get_vector_from_text(self, text):
        # Implement the conversion from text to vector (replace the placeholder)
        # Example: Use a pre-trained language model for feature extraction
        return np.random.rand(self.dimension).astype('float32')

    def get_text_from_index(self, index):
        # Implement the method to retrieve text associated with the given index
        # This will depend on how you have stored or associated text with each vector index
        return "Text for Index {}".format(index)

    def get_vectors_from_texts(self, texts):
        return [self.get_vector_from_text(text) for text in texts]
