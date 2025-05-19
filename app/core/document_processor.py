import os
import re
from typing import List, Optional
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma

from app.core.model_utils import get_embeddings
from app.config import DOCUMENTS_DIR, VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP
import app.core.shared_instances as shared


def load_documents(directory_path: str = DOCUMENTS_DIR) -> List[Document]:
    documents = []
    directory = Path(directory_path)
    os.makedirs(directory, exist_ok=True)

    for file_path in directory.iterdir():
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())
            elif file_path.suffix.lower() == ".csv":
                loader = CSVLoader(str(file_path))
                documents.extend(loader.load())
            elif file_path.suffix.lower() == ".txt":
                loader = TextLoader(str(file_path), encoding="utf-8")
                documents.extend(loader.load())
        except Exception as e:
            print("Unexpected error:", e)
            pass

    return documents


def process_documents(documents: List[Document], debug: bool = True) -> List[Document]:
    """
    split documents into chunks of text with a maximum size of CHUNK_SIZE
    """
    if not documents:
        return []

    chunks = []
    sentence_endings = r"([。！？；.!?;])"

    sorted_docs = sorted(
        documents,
        key=lambda doc: (doc.metadata.get("source", ""), doc.metadata.get("page", 0)),
    )

    last_incomplete_sentence = ""
    last_doc_source = None
    last_doc_page = -1

    for doc in sorted_docs:
        current_source = doc.metadata.get("source", "")
        current_page = doc.metadata.get("page", 0)
        text = doc.page_content

        text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)

        # handle spaces between Chinese characters
        old_text = ""
        while old_text != text:
            old_text = text
            text = re.sub(r"([\u4e00-\u9fff])\s+([\u4e00-\u9fff])", r"\1\2", text)

        # handle spaces between English words
        text = re.sub(r" {2,}", lambda m: m.group(0)[1:], text)
        is_continuation = (
            last_doc_source == current_source
            and last_doc_page == current_page - 1
            and last_incomplete_sentence
        )

        parts = re.split(sentence_endings, text)

        if is_continuation:
            if parts and not re.match(sentence_endings, parts[0]):
                parts[0] = last_incomplete_sentence + parts[0]
                last_incomplete_sentence = ""

        sentences = []
        i = 0
        while i < len(parts) - 1:
            if i + 1 < len(parts):
                sentence = parts[i] + parts[i + 1]
                sentences.append(sentence)
            i += 2

        if i < len(parts) and parts[i].strip():
            last_incomplete_sentence = parts[i]
        else:
            last_incomplete_sentence = ""

        last_doc_source = current_source
        last_doc_page = current_page

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if sentence_length > CHUNK_SIZE:
                if current_chunk:
                    chunks.append(
                        Document(
                            page_content="".join(current_chunk), metadata=doc.metadata
                        )
                    )
                    current_chunk = []
                    current_length = 0

                chunks.append(Document(page_content=sentence, metadata=doc.metadata))
                continue

            if current_length + sentence_length > CHUNK_SIZE and current_chunk:
                chunks.append(
                    Document(page_content="".join(current_chunk), metadata=doc.metadata)
                )
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        if current_chunk:
            chunks.append(
                Document(page_content="".join(current_chunk), metadata=doc.metadata)
            )

    if last_incomplete_sentence and len(last_incomplete_sentence) <= CHUNK_SIZE:
        if sorted_docs:
            last_doc = sorted_docs[-1]
            chunks.append(
                Document(
                    page_content=last_incomplete_sentence, metadata=last_doc.metadata
                )
            )
    if debug:
        print(f"generated {len(chunks)} chunks")
        if chunks:
            for i in range(16):
                print(f"chunk #{i+1} (length {len(chunks[i].page_content)}):")
                print(chunks[i].page_content)
        else:
            print("chunks generating failed")

    return chunks


def create_vector_store(
    chunks: List[Document], embedding_model=None
) -> Optional[Chroma]:
    """
    Create a vector store from a list of documents.
    """
    if not chunks:
        return None

    try:
        embeddings = get_embeddings()
        if embeddings is None:
            return None

        # Ensure the directory exists
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)

        # Create or update the vector store
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH,
        )
        return vector_db

    except Exception as e:
        print(f"Error creating vector store: {str(e)}")
        return None


def get_vector_store(force_reload: bool = False) -> Optional[Chroma]:
    if shared.vector_db is not None and not force_reload:
        return shared.vector_db
    try:
        embedding_model = get_embeddings()
        if embedding_model is None:
            print("cannot load embedding model")
            return None

        if os.path.exists(VECTOR_DB_PATH):
            shared.vector_db = Chroma(
                persist_directory=VECTOR_DB_PATH, embedding_function=embedding_model
            )
            return shared.vector_db
        else:
            return None
    except Exception as e:
        print(f"Vectordb loading error: {str(e)}")
        return None


def build_knowledge_base(documents_dir: str = DOCUMENTS_DIR) -> bool:
    try:
        documents = load_documents(documents_dir)
        if not documents:
            return False
        chunks = process_documents(documents)

        if not chunks:
            return False
        embedding_model = get_embeddings()

        if not embedding_model:
            return False

        vector_db = create_vector_store(chunks, embedding_model)
        vector_db_exists = vector_db is not None
        print(vector_db_exists)
        return vector_db_exists
    except Exception as e:
        print("Vectordb building error:", e)
        return False
