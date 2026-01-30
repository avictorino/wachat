# messaging/tasks.py
import logging

from messaging.types import IncomingMessage
from whatsapp import FacebookWhatsAppProvider, handle_incoming_message

logger = logging.getLogger(__name__)


def process_message_task(incoming_message: IncomingMessage) -> None:
    try:

        outgoing = handle_incoming_message(incoming_message)
        provider = FacebookWhatsAppProvider.from_settings()
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
