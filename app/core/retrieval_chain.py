from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever

from app.config import SEARCH_K
import app.core.shared_instances as shared
from app.core.model_utils import get_llm
from app.core.document_processor import get_vector_store


FINANCE_QA_PROMPT_TEMPLATE = """  
你是一位专业的金融专家顾问，拥有丰富的理论基础和经验，了解金融领域的各种知识。   
基于以下丰富的已知信息，专业而有条理地回答问题。
如果已知信息不足，可以提供一般性的相关知识，但请明确说明这是通用知识而非来自文档。  
已知信息: {context}  
用户问题: {question}  
请以有逻辑、清晰的方式回答问题，依据提供的背景信息。
回答中应注重专业性和实用性，避免过度技术性的术语，确保普通人也能理解。  
所有回答必须使用简体中文，并保持专业性、准确性和可读性。
直接回答问题，无需重复问题。  
"""


def create_qa_chain() -> RetrievalQA:
    try:
        if shared.qa_chain is not None:
            return shared.qa_chain

        llm_instance = get_llm()
        if llm_instance is None:
            return None

        vector_db = get_vector_store()
        if vector_db is None:
            return None

        bm25_retriever = vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": SEARCH_K}
        )
        semantic_retriever = vector_db.as_retriever(
            search_type="mmr", search_kwargs={"k": SEARCH_K}
        )

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5]
        )

        prompt = PromptTemplate(
            template=FINANCE_QA_PROMPT_TEMPLATE, input_variables=["context", "question"]
        )

        shared.qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=ensemble_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

        return shared.qa_chain

    except Exception as e:
        raise RuntimeError(f"cannot create QA chain: {str(e)}")
