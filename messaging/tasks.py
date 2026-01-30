# messaging/tasks.py
import logging

from messaging.providers.factory import get_provider
from messaging.services.chat_service import handle_incoming_message
from messaging.types import IncomingMessage

logger = logging.getLogger(__name__)


def process_message_task(incoming_message: IncomingMessage) -> None:
    try:

        outgoing = handle_incoming_message(incoming_message)
        provider = get_provider(outgoing.channel)
        provider.send(outgoing)

    except Exception as ex:
        logger.exception(
            "Error processing message task",
            extra={
                "task": "process_message_task",
                "exception_type": type(ex).__name__,
                "exception_message": str(ex),
            },
        )
