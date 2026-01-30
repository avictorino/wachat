import json
import random

from django.contrib.auth.models import User

from biblical_friend.constants import biblical_names, Gender
from biblical_friend.llm.base import LLMMessage
from biblical_friend.llm.factory import get_llm_client
from biblical_friend.models import UserSpiritualProfile, VirtualFriend
from biblical_friend.services.orchestrator import chat_with_friend
from biblical_friend.services.prompt_builder import build_gender_inference_prompt
from messaging.types import IncomingMessage, OutgoingMessage


def extract_whatsapp_identity(payload: dict) -> dict:
    return {
        "channel": "whatsapp",
        "wa_id": payload.get("WaId"),
        "from": payload.get("From"),
        "to": payload.get("To"),
    }

def handle_incoming_message(msg: IncomingMessage) -> OutgoingMessage:

    identity = extract_whatsapp_identity(msg.raw_payload)

    friend = get_friend_or_init_person(msg)

    result, conversation = chat_with_friend(
        friend=friend,
        user_text=msg.text or "",
        llm=get_llm_client(),
        identity=identity
    )

    return OutgoingMessage(
        channel=msg.channel,
        from_=msg.to,              # 👈 invertendo
        to=msg.from_,  # 👈 invertendo
        text=result.text,
        reply_as_audio=msg.reply_as_audio
    )


def get_friend_or_init_person(msg: IncomingMessage) -> VirtualFriend:

    user = User.objects.filter(username=msg.from_.replace("whatsapp:", "")).first()
    names = biblical_names
    if not user and msg.raw_payload:
        metadata = msg.raw_payload.get("ChannelMetadata")
        first_name = None
        last_name = None
        if metadata and isinstance(metadata, list) and len(metadata) > 0:
            metadata = metadata[0]
            metadata = json.loads(metadata)
            name = metadata.get("data", {}).get("context", {}).get("ProfileName", {})

            if name:
                first_name = name.split(" ")[0]
                last_name = name.split(" ")[-1]

        user, created = User.objects.get_or_create(
            username=msg.from_.replace("whatsapp:", ""),
            first_name=first_name,
            last_name=last_name,
            is_active=True
        )

        if created:
            gender_found = infer_gender_from_name(
                name=first_name,
                country="Brasil",
            )

            UserSpiritualProfile.objects.create(user=user, gender=gender_found)

            names = [
                b for b in biblical_names
                if b["gender"] == gender_found
            ]

    friend_name = random.choice(names)
    friend, _ = VirtualFriend.objects.get_or_create(
        owner=user,
        defaults={
            "name": friend_name.get("name"),
            "gender": friend_name.get("gender"),
        }
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