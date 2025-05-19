import re
from flask_cors import CORS
import os
import sys
import json
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    Response,
    stream_with_context,
)

from app.config import SEARCH_K

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
CORS(app)

global_qa_chain = None
global_vector_db = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query/stream", methods=["POST", "GET"])
def stream_query_knowledge_base():
    """Stream processing of knowledge base queries"""
    if not global_qa_chain:
        return jsonify(
            {
                "status": "error",
                "message": "The knowledge base is not initialized. Please build it first",
            }
        )

    if request.method == "POST":
        data = request.get_json()
        query = data.get("query", "")
    else:
        query = request.args.get("q", "")

    if not query.strip():
        return jsonify({"status": "error", "message": "Query content cannot be empty"})

    def is_significant_overlap(text1, text2, overlap_threshold=0.4):
        norm_text1 = re.sub(r"\s+", "", text1).lower()
        norm_text2 = re.sub(r"\s+", "", text2).lower()

        shorter = norm_text1 if len(norm_text1) <= len(norm_text2) else norm_text2
        longer = norm_text2 if len(norm_text1) <= len(norm_text2) else norm_text1

        max_match_length = 0
        for i in range(len(shorter)):
            for j in range(i + 1, len(shorter) + 1):
                substring = shorter[i:j]
                if len(substring) > 30 and substring in longer:
                    max_match_length = max(max_match_length, len(substring))

        overlap_ratio = max_match_length / len(shorter)
        return overlap_ratio >= overlap_threshold

    def generate():
        from app.core.model_utils import get_llm
        from app.core.document_processor import get_vector_store

        try:
            # Get relevant documents
            vector_db = get_vector_store()
            docs = vector_db.similarity_search(query, k=SEARCH_K)

            # Build context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Get LLM with streaming callback
            streaming_llm = get_llm(streaming=True)

            # Build prompt
            from app.core.retrieval_chain import FINANCE_QA_PROMPT_TEMPLATE
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate(
                template=FINANCE_QA_PROMPT_TEMPLATE,
                input_variables=["context", "question"],
            )

            # Prepare source document information but do not send yet - modified for content deduplication
            sources = []
            has_significant_overlap = False

            for doc in docs:
                content = doc.page_content.strip()
                source_name = (
                    doc.metadata.get("source", "unknown")
                    if hasattr(doc, "metadata")
                    else "unknown"
                )

                has_significant_overlap = False
                for existing_source in sources:
                    if is_significant_overlap(content, existing_source["content"]):
                        has_significant_overlap = True
                        break

                if not has_significant_overlap:
                    source = {"content": content, "source": source_name}
                    sources.append(source)

            full_prompt = prompt.format(context=context, question=query)

            # Start streaming generation - use callback for streaming processing
            for chunk in streaming_llm.stream(full_prompt):

                if hasattr(chunk, 'content'):  # AIMessageChunk
                    token_content = chunk.content
                else:  # string
                    token_content = chunk
                    
                yield f"data: {json.dumps({'type': 'token', 'token': token_content})}\n\n"

            # After text generation completes, send source documents information
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        except Exception as e:
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/rebuild", methods=["POST"])
def rebuild_knowledge_base():
    global global_qa_chain, global_vector_db

    try:
        from app.core.document_processor import build_knowledge_base
        from app.core.retrieval_chain import create_qa_chain

        success = build_knowledge_base()
        if success:
            global_qa_chain = create_qa_chain()
            return jsonify(
                {
                    "status": "success",
                    "message": "The knowledge base was rebuilt successfully",
                }
            )
        else:
            return jsonify(
                {
                    "status": "error",
                    "message": "Failed to rebuild the knowledge base",
                }
            )
    except Exception as e:
        return jsonify(
            {
                "status": "error",
                "message": f"Failed to rebuild the knowledge base: {str(e)}",
            }
        )


@app.route("/api/documents", methods=["GET"])
def get_documents():
    try:
        from app.config import DOCUMENTS_DIR

        if not os.path.exists(DOCUMENTS_DIR):
            return jsonify({"status": "success", "documents": []})

        documents = []
        for file_name in os.listdir(DOCUMENTS_DIR):
            file_path = os.path.join(DOCUMENTS_DIR, file_name)
            if os.path.isfile(file_path) and file_name != ".gitkeep":
                documents.append(file_name)

        return jsonify({"status": "success", "documents": documents})
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"cannot get documents: {str(e)}"}
        )


@app.route("/api/info", methods=["GET"])
def get_system_info():
    try:
        
        from app.config import (
            LOCAL_LLM_MODEL,
            EMBEDDING_MODEL,
            MODEL_NUM_THREAD,
            USE_OPENAI,
            OPENAI_API_KEY,
            OPENAI_LLM_MODEL,
        )

        if USE_OPENAI and OPENAI_API_KEY:
            model = OPENAI_LLM_MODEL
        else:
            model = LOCAL_LLM_MODEL
        return jsonify(
            {
                "status": "success",
                "model": model,
                "embedding_model": EMBEDDING_MODEL,
                "threads": MODEL_NUM_THREAD,
                "initialized": global_qa_chain is not None,
            }
        )
    except Exception as e:
        return jsonify(
            {"status": "error", "message": f"cannot get system info: {str(e)}"}
        )


def init_app():
    global global_qa_chain, global_vector_db

    try:
        from app.core.document_processor import get_vector_store
        from app.core.retrieval_chain import create_qa_chain
        from app.core.model_utils import get_embeddings

        embeddings = get_embeddings()
        if not embeddings:
            print(
                "Unable to load the embedding model. The application may not function properly."
            )
            return

        global_vector_db = get_vector_store(embeddings)
        if global_vector_db:
            global_qa_chain = create_qa_chain()
        else:
            global_qa_chain = None

    except Exception as e:
        print(f"initializing error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch the Knowledge Base Web Application"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port for the web service"
    )
    parser.add_argument(  
        "--use-openai",   
        action="store_true",  
        help="Use OpenAI API instead of local model"  
    ) 
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="BAAI/bge-base-zh-v1.5",
        help="embedding model",
    )
    parser.add_argument("--num-threads", type=int, default=12, help="threads number")
    args = parser.parse_args()

    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    os.environ["MODEL_NUM_THREAD"] = str(args.num_threads)
    os.environ["USE_OPENAI"] = str(args.use_openai)

    init_app()

    print(f"Starting web service at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
