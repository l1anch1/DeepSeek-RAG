from app.config import LOCAL_LLM_MODEL, OPENAI_API_KEY, OPENAI_LLM_MODEL, OPENAI_API_BASE, USE_OPENAI


HELLO_PROMPT = "Who are you?"  


def test_model_local(model=LOCAL_LLM_MODEL):  
    from langchain_ollama import OllamaLLM 
    try:  
        llm = OllamaLLM(model=model, streaming=True)  
        
        for chunk in llm.stream(HELLO_PROMPT):  
            print(chunk, end="", flush=True)  
        print()  
        
    except Exception as e:  
        print(f"model calling error: {str(e)}")  


def test_model_openai():
    try:  
        from langchain_openai import ChatOpenAI  
        llm = ChatOpenAI(model=OPENAI_LLM_MODEL, streaming=True, openai_api_key=OPENAI_API_KEY, 
                         openai_api_base=OPENAI_API_BASE)  

        for chunk in llm.stream(HELLO_PROMPT):
            if hasattr(chunk, 'content'):
                print(chunk.content, end="", flush=True)
            else:
                print(chunk, end="", flush=True)
        print()  
        
    except Exception as e:  
        print(f"model calling error: {str(e)}")
if __name__ == "__main__":  
    if USE_OPENAI:  
        print(f"Testing OpenAI model: {OPENAI_LLM_MODEL}")
        test_model_openai()
        
    else:
        print(f"Testing local model: {LOCAL_LLM_MODEL}")
        test_model_local()  