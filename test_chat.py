import pandas as pd
from components.llm_chat import DataChat

def test_chat():
    # Mock data
    data = {'id': [1, 2, 3], 'col1': ['A', 'B', 'C'], 'col2': [10.5, 20.1, 15.3]}
    df = pd.DataFrame(data)
    
    # Mock profile and anomalies
    profile = {
        'overview': {'n_rows': 3, 'n_cols': 3},
        'columns': {
            'id': {'dtype': 'int64', 'unique': 3, 'missing': 0},
            'col1': {'dtype': 'object', 'unique': 3, 'missing': 0},
            'col2': {'dtype': 'float64', 'unique': 3, 'missing': 0}
        }
    }
    anomalies = {'summary': {'total_anomalies': 0}}
    
    print("Testing DataChat...")
    # Allow user to specify model if they want, default to 'mistral' or 'llama3'
    # For test, we'll try 'mistral' but catch if it fails
    chat = DataChat(df, profile, anomalies, model_name="gemma3:4b") 
    
    print("\n--- Building Context ---")
    context = chat.build_context_prompt()
    print("Context (first 200 chars):")
    print(context[:200] + "...")
    
    print("\n--- Asking Question ---")
    question = "What is the average of col2?"
    print(f"Question: {question}")
    
    response = chat.ask(question)
    print("\nResponse:")
    print(response['answer'])
    
    if "Error" in response['answer']:
        print("\n[NOTE] If this error is about connection, make sure 'ollama serve' is running.")
        print("[NOTE] If it's about model not found, run 'ollama pull mistral'.")


if __name__ == "__main__":
    test_chat()
