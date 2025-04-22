def load_data_from_txt(file_path):
    data_pairs = []  # To store questions and answers
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    question = None
    answer = None
    for line in lines:
        if line.startswith("Question:"):
            question = line.replace("Question:", "").strip()
        elif line.startswith("Answer:"):
            answer = line.replace("Answer:", "").strip()
            if question and answer:
                data_pairs.append((question, answer))  # Store as a pair
                question, answer = None, None
    return data_pairs


# Load the file
training_data = load_data_from_txt("data.txt")
print(training_data)  # Example Output: [('What is LangChain?', 'LangChain is a...'), ...]

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI


# Create a dynamic prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer based on the provided data."),
        *[("user", f"Question: {q}\nAnswer: {a}") for q, a in training_data],
        ("user", "Question: {question}"),  # Placeholder for user queries
    ]
)

# Initialize the LLM
llm = Ollama(model="llama2")  # Replace with your model
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Example query
query = "What is LangChain?"
response = chain.invoke({"question": query})
print("Answer:", response)

def save_to_txt(file_path, question, answer):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"\nQuestion: {question}\nAnswer: {answer}")


# Example usage:
new_question = "What is Python?"
new_answer = "Python is a programming language used for various applications."

save_to_txt("data.txt", new_question, new_answer)
print("New data saved!")
