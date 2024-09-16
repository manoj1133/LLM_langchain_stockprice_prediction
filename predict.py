import numpy as np
from langchain_setup import setup_langchain

def format_data_for_gpt4(sequences):
    formatted_data = []
    for seq in sequences:
        prompt = " ".join([str(x[0]) for x in seq])
        formatted_data.append(prompt)
    return formatted_data

def generate_predictions(formatted_data, api_key):
    llm_chain = setup_langchain(api_key)
    predictions = []
    for prompt in formatted_data:
        result = llm_chain.run(sequence=prompt)
        predictions.append(float(result.strip()))
    return predictions

if __name__ == "__main__":
    print("This file is not meant to be run directly. Please run main.py.")
