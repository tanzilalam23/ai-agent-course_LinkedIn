from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import pipeline

# 1. Load .env file (contains your HF_TOKEN)
load_dotenv()

# 2. Log in to Hugging Face with your token
hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... in your .env file.")
login(token=hf_token)

# 3. Choose a small, free text-generation model
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# 4. Create a local text-generation pipeline
pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID
)

print("✅ Agent ready! Type a question, or 'quit' to stop.\n")

while True:
    try:
        user_input = input("> ").strip()

        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_input:
            continue


        # 5. Use chat-style messages for an instruction-tuned chat model
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Answer clearly and briefly."
            },
            {
                "role": "user",
                "content": user_input
            }
        ]


        # 6. Generate the response locally
        result = pipe(
            messages,
            max_new_tokens=100,
            do_sample=False,
            repetition_penalty=1.1
        )[0]["generated_text"]


        # 7. Extract the assistant's reply
        answer = result[-1]["content"]


        print(f"\n🤖 Agent: {answer}\n")


    except KeyboardInterrupt:
        print("\n(Stopped) Goodbye!")
        break
