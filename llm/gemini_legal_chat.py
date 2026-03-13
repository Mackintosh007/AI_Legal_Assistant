import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from retrieval.retriever import LegalRetriever

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


print("Loading legal retriever...")
retriever = LegalRetriever()


print("Loading reasoning model (Phi-3 Mini)...")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)


conversation_history = []


def build_context(results):

    context = ""

    for r in results:

        context += f"""
[Legal Source]

Act: {r['act']}
Section: {r['section']}

Provision:
{r['text']}

---
"""

    return context


def build_history(history):

    history_text = ""

    for h in history[-3:]:
        history_text += f"""
User: {h['question']}
Assistant: {h['answer']}
"""

    return history_text


def ask_legal_ai(question, history):

    results = retriever.search(question)

    context = build_context(results)

    history_text = build_history(history)

    prompt = f"""
You are an expert Nigerian legal assistant.

Use the legal sources to answer the question.

Follow these steps:

1. Identify the legal rule
2. Explain the rule
3. Give a simple explanation for citizens
4. Explain limitations of the law
5. Cite the Act and Section

Conversation History:
{history_text}

Legal Sources:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=350,
        temperature=0.2
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    history.append({
        "question": question,
        "answer": response
    })

    return response, results


def main():

    print("\nNigerian Legal AI Ready.\n")

    while True:

        try:

            question = input("\nAsk Nigerian Legal AI: ")

            answer, sources = ask_legal_ai(question, conversation_history)

            print("\nAnswer:\n")
            print(answer)

            print("\nSources:\n")

            for s in sources:
                print(f"{s['act']} — Section {s['section']}")

        except KeyboardInterrupt:

            print("\nExiting Legal AI.")
            break


if __name__ == "__main__":
    main()