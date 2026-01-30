from __future__ import annotations

from datetime import timedelta

from django.utils import timezone
from django.utils.dateparse import parse_datetime

from core.models import Conversation, Message, VirtualFriend


def is_whatsapp_window_open(conversation: Conversation) -> bool:
    last_user_msg = conversation.context.get("last_user_message_at")
    if not last_user_msg:
        return False

    return timezone.now() - parse_datetime(last_user_msg) < timedelta(hours=24)


def get_or_create_open_conversation(
    friend: VirtualFriend,
    channel: str,
    channel_user_id: str,
    title: str = "",
) -> Conversation:

    convo = (
        Conversation.objects.filter(
            friend=friend,
            is_closed=False,
            context__channel=channel,
            context__channel_user_id=channel_user_id,
        )
        .order_by("-created_at")
        .first()
    )

    if convo:
        if not is_whatsapp_window_open(convo):
            convo.is_closed = True
            convo.save()
        else:
            return convo

    return Conversation.objects.create(
        friend=friend,
        title=title,
        context={
            "channel": channel,
            "channel_user_id": channel_user_id,
            "last_user_message_at": timezone.now().isoformat(),
        },
    )


def get_recent_messages(conversation: Conversation, limit: int = 20) -> list[Message]:
    return list(conversation.messages.order_by("-created_at")[:limit][::-1])
