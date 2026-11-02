"""Model 4: Kronos forecast paper bots shown at /model4. See model4/bot.py."""
import os


def attach(app):
    """Serve /model4 on `app`, and start the bot where the other bots run (PAPER_BOT) or MODEL4_BOT is set."""
    from model4.bot import router, start_thread

    app.include_router(router)
    if os.environ.get("PAPER_BOT") or os.environ.get("MODEL4_BOT"):
        start_thread()
