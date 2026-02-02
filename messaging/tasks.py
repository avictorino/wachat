# messaging/tasks.py
import logging

from messaging.types import (
    IncomingMessage,
    CHANNEL_TELEGRAM,
)

logger = logging.getLogger(__name__)


def get_provider_for_channel(channel: str):
    """
    Get the appropriate provider instance for a given channel.

    Args:
        channel: The channel type (telegram)

    Returns:
        Provider instance for sending messages

    Raises:
        ValueError: If channel is not supported or provider cannot be initialized
    """
    if channel == CHANNEL_TELEGRAM:
        from service.telegram import TelegramProvider

        return TelegramProvider.from_settings()
    else:
        raise ValueError(f"Unsupported channel: {channel}")


def process_message_task(incoming_message: IncomingMessage) -> None:
    try:
        from service.telegram import handle_incoming_message

        outgoing = handle_incoming_message(incoming_message)
        provider = get_provider_for_channel(outgoing.channel)
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
