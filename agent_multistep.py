from dotenv import load_dotenv
from transformers import pipeline
import os
import re




# ---------------------------
# Setup (same as we did before)
# ---------------------------
load_dotenv()
token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise SystemExit("❌ No Hugging Face token found. Add HF_TOKEN=... to your .env")






MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

pipe = pipeline(
    task="text-generation",
    model=MODEL_ID,
    tokenizer=MODEL_ID,
    device=-1,              # CPU for stability/consistency
)





# ---------------------------
# Tiny helpers (post-process)
# ---------------------------
STOPWORDS = {
    "the","a","an","and","or","for","to","of","in","on","with","at","by","from","is","are",
    "this","that","those","these","about","into","as","it","its","be","being","been","was","were"
}

def one_line(text: str) -> str:
    return text.strip().splitlines()[0].strip()

def clip(text: str, max_chars: int = 160) -> str:
    t = one_line(text)
    return (t[:max_chars].rstrip(" ,.;:!-")) if len(t) > max_chars else t

def title_case_five_words(text: str) -> str:
    # Keep only letters/numbers, split, take 5 non-stopwords if possible.
    words = re.findall(r"[A-Za-z0-9]+", text)
    if not words:
        return "Short Title Placeholder"
    # prefer non-stopwords first
    content = [w for w in words if w.lower() not in STOPWORDS] or words
    trimmed = content[:5] if len(content) >= 5 else (content + ["Ideas"] * (5 - len(content)))
    return " ".join(w.capitalize() for w in trimmed[:5])

def three_hashtags(summary: str, topic: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", f"{topic} {summary}")
    # prefer meaningful tokens
    candidates = []
    seen = set()
    for w in words:
        wl = w.lower()
        if wl in STOPWORDS or len(wl) < 3:
            continue
        if wl not in seen:
            seen.add(wl)
            candidates.append(wl)
    # fallback if too few
    if len(candidates) < 3:
        candidates.extend(["ai", "learning", "topic"])
    tags = [f"#{candidates[i][:24]}" for i in range(3)]
    return " ".join(tags)





# ---------------------------
# Core multi-step logic
# ---------------------------
def ask(prompt: str, max_new_tokens: int = 60) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Follow the user's instructions exactly."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    result = pipe(
        messages,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=1.1,

    )

    return result[0]["generated_text"][-1]["content"].strip()

def process_topic(topic: str):
    # Step 1: summary
    p1 = (
        "Write one short sentence of no more than 18 words summarizing this topic "
        "for a general audience. Return only the sentence.\n"
        f"Topic: {topic}"
    )
    summary = clip(ask(p1, max_new_tokens=50), 140)

    # Step 2: title
    p2 = (
        "Write a catchy title of exactly five words for this summary. "
        "Return only the title. No punctuation or quotes.\n"
        f"Summary: {summary}"
    )
    raw_title = ask(p2, max_new_tokens=50)
    title = title_case_five_words(raw_title)

    if title.lower() in summary.lower() or len(title.split()) != 5:
        title = title_case_five_words(summary)

    # Step 3: hashtags
    p3 = (
        "Suggest exactly three short hashtags for this topic. "
        "Return only the hashtags separated by single spaces.\n"
        f"Summary: {summary}"
    )
    raw_tags = ask(p3, max_new_tokens=50)

    found = re.findall(r"#\w+", raw_tags)
    hashtags = " ".join(found[:3]) if len(found) == 3 else three_hashtags(summary, topic)

    return summary, title, hashtags

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print("🧠 Multi-Step Agent ready! Type a topic, or 'quit' to stop.\n")
    while True:
        topic = input("Enter a topic: ").strip()
        if topic.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if not topic:
            continue

        print("\n--- Thinking in steps ---\n")
        summary, title, hashtags = process_topic(topic)
        print("✅ Final Output:")
        print(f"Summary: {summary}")
        print(f"Title: {title}")
        print(f"Hashtags: {hashtags}\n")