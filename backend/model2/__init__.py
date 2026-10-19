"""Model 2: self-learning paper bots shown at /model2. See model2/agent.py."""
import os


def attach(app):
    """Serve /model2 on `app`, and start the bots where model 1's bots run (PAPER_BOT) or MODEL2_BOT is set."""
    from model2.bot import router, start_threads

    app.include_router(router)
    if os.environ.get("PAPER_BOT") or os.environ.get("MODEL2_BOT"):
        start_threads()
