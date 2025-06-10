import os
from app import process_document, add_to_vector_collection
from streamlit.runtime.uploaded_file_manager import UploadedFile

DOCS_DIR = "docs"

class DummyUploadedFile:
    """A dummy class to mimic Streamlit's UploadedFile for local files."""
    def __init__(self, path):
        self.name = os.path.basename(path)
        self._file = open(path, "rb")
    def read(self):
        self._file.seek(0)
        return self._file.read()
    def close(self):
        self._file.close()

def main():
    for fname in os.listdir(DOCS_DIR):
        if fname.lower().endswith(".pdf"):
            fpath = os.path.join(DOCS_DIR, fname)
            print(f"Processing {fpath} ...")
            uploaded_file = DummyUploadedFile(fpath)
            splits = process_document(uploaded_file)
            normalized_name = fname.translate(str.maketrans({"-": "_", ".": "_", " ": "_"}))
            add_to_vector_collection(splits, normalized_name)
            uploaded_file.close()
            print(f"Added {fname} to vector DB.")

if __name__ == "__main__":
    main()
