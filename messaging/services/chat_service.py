import json
import random

from django.contrib.auth.models import User

from core.constants import biblical_names
from core.llm.base import LLMMessage
from core.llm.factory import get_llm_client
from core.models import UserSpiritualProfile, VirtualFriend
from core.services.orchestrator import chat_with_friend
from core.services.prompt_builder import build_gender_inference_prompt
from messaging.types import IncomingMessage, OutgoingMessage


def extract_whatsapp_identity(payload: dict) -> dict:
    # Facebook WhatsApp API format
    # For Facebook, we extract from the webhook payload structure
    if not payload:
        return {
            "channel": "whatsapp_facebook",
            "wa_id": None,
            "from": None,
            "to": None,
        }

    # Try to extract from Facebook format
    entry = payload.get("entry", [{}])[0]
    changes = entry.get("changes", [{}])[0]
    value = changes.get("value", {})
    messages = value.get("messages", [{}])[0]

    return {
        "channel": "whatsapp_facebook",
        "wa_id": messages.get("from"),
        "from": messages.get("from"),
        "to": value.get("metadata", {}).get("phone_number_id"),
    }


def handle_incoming_message(msg: IncomingMessage) -> OutgoingMessage:

    identity = extract_whatsapp_identity(msg.raw_payload)

    friend = get_friend_or_init_person(msg)

    result, conversation = chat_with_friend(
        friend=friend, user_text=msg.text or "", llm=get_llm_client(), identity=identity
    )

    return OutgoingMessage(
        channel=msg.channel,
        from_=msg.to,  # 👈 invertendo
        to=msg.from_,  # 👈 invertendo
        text=result.text,
        reply_as_audio=msg.reply_as_audio,
    )


def get_friend_or_init_person(msg: IncomingMessage) -> VirtualFriend:

    user = User.objects.filter(username=msg.from_).first()
    names = biblical_names
    if not user and msg.raw_payload:
        # Try to extract profile name from Facebook WhatsApp webhook
        first_name = None
        last_name = None

        # Facebook WhatsApp format
        try:
            entry = msg.raw_payload.get("entry", [{}])[0]
            changes = entry.get("changes", [{}])[0]
            value = changes.get("value", {})
            contacts = value.get("contacts", [])
            if contacts:
                profile = contacts[0].get("profile", {})
                name = profile.get("name", "")
                if name:
                    name_parts = name.split(" ")
                    first_name = name_parts[0]
                    last_name = name_parts[-1] if len(name_parts) > 1 else name_parts[0]
        except (KeyError, IndexError, AttributeError):
            pass

        user, created = User.objects.get_or_create(
            username=msg.from_,
            first_name=first_name,
            last_name=last_name,
            is_active=True,
        )

        if created:
            gender_found = infer_gender_from_name(
                name=first_name,
                country="Brasil",
            )

            UserSpiritualProfile.objects.create(user=user, gender=gender_found)

            names = [b for b in biblical_names if b["gender"] == gender_found]

    friend_name = random.choice(names)
    friend, _ = VirtualFriend.objects.get_or_create(
        owner=user,
        defaults={
            "name": friend_name.get("name"),
            "gender": friend_name.get("gender"),
        },
    )

    return friend


def infer_gender_from_name(*, name: str, country: str) -> str:
    """
    Returns: 'male', 'female', or 'unknown'
    """
    if not name:
        return "unknown"

    llm = get_llm_client()
    prompt = build_gender_inference_prompt(
        profile_name=name,
        country=country,
    )

    resp = llm.chat(
        messages=[LLMMessage(role="system", content=prompt)],
        temperature=0.0,
    )

    try:
        data = json.loads(resp.text)
        gender = data.get("gender", "unknown")
    except json.JSONDecodeError:
        return "unknown"

    return gender if gender in {"male", "female", "unknown"} else "unknown"
