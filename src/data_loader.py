from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader


def load_all_documents(dir: str) -> List[Any]:
    """Load all supported file from the given directory and conver to
    Langchain document structure

    Args:
      dir: path to the folder
    """
    BASE_DIR = Path(__file__).parent
    data_path = (
        BASE_DIR / dir
    ).resolve()  # resolve is for getting absalute cleaned path
    print(f"[DEBUG] Data path: {data_path}")

    documents = []

    # pdf files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(
        f"[DEBUG] Found {len(pdf_files)} files, PDF files: {[file.name for file in pdf_files]}"
    )

    for file in pdf_files:
        print(f"[DEBUG] Loading file: {file}")

        try:
            pdf_loader = PyMuPDFLoader(str(file))
            loaded = pdf_loader.load()
            documents.extend(loaded)
            print(f"[DEBUG] Loaded {len(loaded)} from {file}")

        except Exception as e:
            print(f"[ERROR]: Failed to lead pdf: {file} : {e}")

    return documents
