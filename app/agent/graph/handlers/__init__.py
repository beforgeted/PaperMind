"""Handler registry mapping intent names to handler node functions."""

from app.agent.graph.handlers.retrieval import retrieval_handler
from app.agent.graph.handlers.profile import profile_handler
from app.agent.graph.handlers.chat import chat_handler
from app.agent.graph.handlers.comparison import comparison_handler
from app.agent.graph.handlers.summary import summary_handler
from app.agent.graph.handlers.writing import writing_handler

HANDLERS = {
    "retrieval_handler": retrieval_handler,
    "profile_handler": profile_handler,
    "chat_handler": chat_handler,
    "comparison_handler": comparison_handler,
    "summary_handler": summary_handler,
    "writing_handler": writing_handler,
}
