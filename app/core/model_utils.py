import os
import glob
import sys
from typing import Optional, Union
from langchain_ollama import OllamaLLM
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain_openai import ChatOpenAI

from app.config import (
    DEVICE,
    LOCAL_LLM_MODEL,
    EMBEDDING_MODEL,
    MODEL_TEMPERATURE,
    MODEL_TOP_P,
    MODEL_NUM_CTX,
    MODEL_NUM_PREDICT,
    MODEL_NUM_THREAD,
    OPENAI_API_KEY,
    OPENAI_LLM_MODEL,
    USE_OPENAI,
    OPENAI_API_BASE,
)
import app.core.shared_instances as shared


def get_llm(
    model: str = LOCAL_LLM_MODEL,
    temperature: float = MODEL_TEMPERATURE,
    top_p: float = MODEL_TOP_P,
    num_ctx: int = MODEL_NUM_CTX,
    num_predict: int = MODEL_NUM_PREDICT,
    streaming: bool = True,
    use_openai: bool = USE_OPENAI,
    
    **kwargs,
) -> Union[OllamaLLM, ChatOpenAI, None]:
    callbacks = []
    if streaming and StreamingStdOutCallbackHandler is not None:
        callbacks.append(StreamingStdOutCallbackHandler())

    try:
        if use_openai:
            # OpenAI model
            if not os.environ.get("OPENAI_API_KEY") and not OPENAI_API_KEY:
                print("Error: OPENAI_API_KEY not set")
                return None
                
            model = OPENAI_LLM_MODEL

            if OPENAI_API_KEY and not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
                
            model_kwargs = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "streaming": streaming,
                "callbacks": callbacks,
                **kwargs,
            }
            if OPENAI_API_BASE:
                model_kwargs["openai_api_base"] = OPENAI_API_BASE

            shared.llm_model = ChatOpenAI(**model_kwargs)
            print(f"loaded openai model: {model}, device: {DEVICE}")
            return shared.llm_model
        else:
            # local model
            model_kwargs = {
                "model": model,
                "temperature": temperature,
                "top_p": top_p,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
                "num_thread": MODEL_NUM_THREAD,
                "callbacks": callbacks,
                **kwargs,
            }

            shared.llm_model = OllamaLLM(**model_kwargs)
            print(f"loaded local model: {model}, device: {DEVICE}")
            return shared.llm_model

    except Exception as e:
        print(f"initializing llm error: {str(e)}")
        return None


def get_embeddings(
    model_name: str = EMBEDDING_MODEL, force_reload=False
) -> Optional[Embeddings]:

    if "/" not in model_name:
        error_msg = (
            f"Error: invalid model name '{model_name}', use org_name/model_name format"
        )
        print(error_msg)
        sys.exit(1)

    if (
        not force_reload
        and shared.embedding_model is not None
        and hasattr(shared, "current_model_name")
        and shared.current_model_name == model_name
    ):
        return shared.embedding_model

    try:
        app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(app_root, "models_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["HF_HOME"] = cache_dir

        org_name, model_short_name = model_name.split("/")
        model_dir_name = f"models--{org_name}--{model_short_name}".replace("/", "--")
        model_base_path = os.path.join(cache_dir, model_dir_name)
        local_model_path = None

        if os.path.exists(model_base_path):
            snapshots_dir = os.path.join(model_base_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshot_dirs = glob.glob(os.path.join(snapshots_dir, "*"))
                for snapshot_dir in snapshot_dirs:
                    config_path = os.path.join(snapshot_dir, "config.json")
                    if os.path.exists(config_path):
                        local_model_path = snapshot_dir
                        print(f"local embedding model: {local_model_path}")
                        break

        model_kwargs = {"device": DEVICE}
        encode_kwargs = {"normalize_embeddings": True}

        if local_model_path:
            shared.embedding_model = HuggingFaceEmbeddings(
                model_name=local_model_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
            )
            print(f"loaded local model: {model_name}, divice: {DEVICE}")
        else:
            print(f"downloading: {model_name}")
            shared.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=cache_dir,
            )

        shared.current_model_name = model_name
        return shared.embedding_model

    except Exception as e:
        print(f"embedding model error: {str(e)}")

    return None
