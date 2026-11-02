"""Model 3: weekly trend rotation paper bot shown at /model3. See model3/strategy.py."""
import os


def attach(app):
    """Serve /model3 on `app`, and start the bot where model 1's bots run (PAPER_BOT) or MODEL3_BOT is set."""
    from model3.bot import router, start_thread

    app.include_router(router)
    if os.environ.get("PAPER_BOT") or os.environ.get("MODEL3_BOT"):
        start_thread()
