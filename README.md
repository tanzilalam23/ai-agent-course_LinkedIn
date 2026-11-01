# Agentic AI Practice — Updated Course Implementation

This repository is an updated version of the hands-on projects from LinkedIn Learning’s  
**Build with AI: Autonomous Agents with LangChain and Hugging Face** by Walter Shields.
Link: https://www.linkedin.com/learning/build-with-ai-autonomous-agents-with-langchain-and-hugging-face/welcome-setup-and-demo-of-your-finished-ai-agent-27491924

The original course used older Hugging Face Transformers APIs, including
`google/flan-t5-base` and the `text2text-generation` pipeline. Current versions of
Transformers no longer support that pipeline in the same way.

This repository updates the course code to work with:

- `Qwen/Qwen2.5-0.5B-Instruct`
- The current `text-generation` pipeline
- Chat-style messages using `system` and `user` roles
- The current assistant-response output format

## What Is Updated

| Original course approach | Updated version |
|---|---|
| `google/flan-t5-base` | `Qwen/Qwen2.5-0.5B-Instruct` |
| `task="text2text-generation"` | `task="text-generation"` |
| Plain-string prompts | Chat messages |
| `result[0]["generated_text"]` | `result[0]["generated_text"][-1]["content"]` |

## How To Use

1. Fork or clone this repository.
2. Create your own `.env` file and add your Hugging Face token:

```env
HF_TOKEN=your_huggingface_token_here
```

3. Install the packages taught in the course.
4. Run any agent script locally.

Example:

```bash
python agent_basic.py
```

## Fork or Clone

If you want to use, modify, or extend this code:

- **Fork** this repository if you want to make your own version.
- **Clone** this repository if you only want to run it locally.

```bash
git clone [https://github.com/tanzilalam23/ai-agent-course_LinkedIn.git](https://github.com/tanzilalam23/ai-agent-course_LinkedIn.git)
cd ai-agent-course_LinkedIn
```

Please keep attribution to the original LinkedIn Learning course and its instructor.

## Credits

Based on the LinkedIn Learning course  
**Build with AI: Autonomous Agents with LangChain and Hugging Face**  
by Walter Shields.

This repository is an independent updated implementation for current Hugging Face
Transformers APIs.