from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

def get_conversational_chain():
    """Load the QA chain with a Google Generative AI model."""
    prompt_template = """
    You are a helpful AI assistant. Your task is to answer questions based solely on the provided context.
    
    Please adhere to the following instructions:
    
    1. **Use only the information provided in the context**. Do not use outside knowledge or assumptions.
    2. **If the answer is not found in the context**, clearly state: "Answer is not available in the context."
    3. **Provide concise and clear answers**. Be as accurate as possible while staying within the information given.
    4. **Do not speculate or invent facts**.
    
    Context:\n{context}\n
    ---
    Question:\n{question}\n
    ---
    **Answer**:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)