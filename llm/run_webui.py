"""Launch the ADVAT LLM chat UI.

    python run_webui.py

Reads settings (and your GROQ_API_KEY) from the environment or from a .env file next to
this script. See llm/.env.example and webui/README.md.
"""
from webui.app import main

if __name__ == "__main__":
    main()
