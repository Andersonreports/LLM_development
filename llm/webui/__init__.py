"""webui: a small chat UI for the ADVAT LLM project.

Two interchangeable backends, chosen at runtime from the browser or with LLM_BACKEND:

  "tinygpt"  our own from-scratch model (../tinygpt), loaded once and kept warm in a
             background thread. Runs entirely on this machine - no data leaves it - but it
             is a character-level *completion* model (it continues your text), not a chat
             assistant, so expect weak, free-associating output until it is scaled up.

  "groq"     Groq Cloud, reached through its OpenAI-compatible API with your GROQ_API_KEY.
             A proper *chat* model (Llama, etc.), fast and genuinely capable - but the
             conversation is sent to Groq's servers, so only use de-identified text.
"""
