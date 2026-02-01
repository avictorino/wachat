"""
Conversation orchestration and management.

This module handles the core conversation logic including memory management,
conversation flow, and message handling.
"""
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Optional, Tuple

from django.db.models import Q
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from core.constants import MODE_PRIORITY, ConversationMode
from core.models import (
    Conversation,
    FriendMemory,
    Message,
    UserSpiritualProfile,
    VirtualFriend,
)
from service.llm import LLMClient, LLMMessage
from service.media_generation import maybe_generate_image
from service.prompts import (
    IMAGE_EXTRACTION_PROMPT,
    INITIAL_WELCOME_MESSAGE,
    build_memory_prompt,
    build_mode_inference_prompt,
    build_onboarding_prompt,
    build_profile_extraction_prompt,
    build_system_prompt,
    generate_first_welcome_message,
    onboarding_question,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Memory Management
# ============================================================================

DEFAULT_MEMORY_KEYS = [
    "favorite_verse",
    "main_struggle",
    "prayer_style",
    "family_context",
]


def get_relevant_memories(friend: VirtualFriend, keys=None) -> list[FriendMemory]:
    if keys is None:
        keys = DEFAULT_MEMORY_KEYS
    qs = (
        FriendMemory.objects.filter(friend=friend, is_active=True)
        .filter(Q(key__in=list(keys)) | Q(kind__in=["prayer", "verse"]))
        .order_by("-updated_at")[:25]
    )
    return list(qs)


def upsert_memory(
    friend: VirtualFriend,
    *,
    kind: str,
    key: str,
    value: str,
    confidence: float = 0.80,
    source: Optional[dict] = None,
) -> FriendMemory:
    obj, _ = FriendMemory.objects.update_or_create(
        friend=friend,
        kind=kind,
        key=key,
        defaults={
            "value": value,
            "confidence": confidence,
            "source": source or {},
            "is_active": True,
        },
    )
    return obj


# ============================================================================
# Conversation Selectors
# ============================================================================


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
    """
    Get or create an open conversation for a friend on a specific channel.
    
    Supports multiple channels:
    - whatsapp_facebook
    - telegram
    - facebook
    - twilio
    - slack
    
    Args:
        friend: The VirtualFriend instance
        channel: The channel type (whatsapp_facebook, telegram, etc.)
        channel_user_id: The user ID on that channel
        title: Optional conversation title
        
    Returns:
        Conversation instance
    """
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


def extract_phone_ddd(phone_number: str) -> str | None:
    """
    Extract Brazilian DDD (area code) from a phone number.
    
    Args:
        phone_number: Phone number string (e.g., "+5521967337683")
        
    Returns:
        DDD as a string (e.g., "21") or None if not found
    """
    if not phone_number:
        return None
    
    # Remove common formatting characters
    clean_phone = phone_number.replace("+", "").replace("-", "").replace(" ", "").replace("(", "").replace(")", "")
    
    # Brazilian numbers start with country code 55 followed by 2-digit DDD
    # Format: 55 + DDD (2 digits) + phone number (8 or 9 digits)
    if clean_phone.startswith("55") and len(clean_phone) >= 4:
        ddd = clean_phone[2:4]
        # Validate DDD is numeric
        if ddd.isdigit():
            return ddd
    
    return None


# ============================================================================
# Core Orchestration
# ============================================================================


@dataclass
class ImageAnalysis:
    should_generate_image: bool = False
    image_type: Optional[str] = None
    visual_elements: List[str] = None
    emotional_tone: Optional[str] = None
    visual_description: Optional[str] = None


@dataclass(frozen=True)
class ChatResult:
    conversation_id: str
    assistant_message_id: str
    text: str
    media_url: Optional[str] = None


def chat_with_friend(
    friend: VirtualFriend, user_text: str, llm: LLMClient, identity: dict
) -> Tuple[ChatResult, Conversation]:
    """
    Main conversation orchestration function.
    
    Handles conversation flow, memory management, and response generation
    for all supported channels.
    
    Args:
        friend: The VirtualFriend instance
        user_text: User's message text
        llm: LLM client for generating responses
        identity: Channel-specific identity information
        
    Returns:
        Tuple of (ChatResult, Conversation)
    """
    from messaging.types import CHANNEL_WHATSAPP_FACEBOOK
    
    conversation = get_or_create_open_conversation(
        friend=friend,
        channel=identity.get("channel", CHANNEL_WHATSAPP_FACEBOOK),
        channel_user_id=identity.get("user_id", identity.get("wa_id")),
    )

    user_msg = Message.objects.create(
        conversation=conversation,
        role=Message.Role.USER,
        content=user_text,
    )

    conversation.context.update(
        {
            "last_user_message_at": timezone.now().isoformat(),
        }
    )

    conversation.save()

    memories = get_relevant_memories(friend)
    assistant_count = conversation.is_onboarding_phase()

    if assistant_count == 0:
        # Generate personalized welcome message
        user_name = friend.owner.first_name or "amigo" if friend.owner.first_name else "amigo"
        
        # Get inferred gender from user's spiritual profile
        inferred_gender = "unknown"
        try:
            profile = friend.owner.spiritual_profile
            inferred_gender = profile.gender if profile.gender else "unknown"
        except Exception:
            pass
        
        # Extract DDD from user's phone number (username is the phone)
        phone_ddd = extract_phone_ddd(friend.owner.username)
        
        # Generate the personalized welcome message
        welcome_text = generate_first_welcome_message(
            user_name=user_name,
            inferred_gender=inferred_gender,
            phone_ddd=phone_ddd,
        )
        
        assistant_msg = Message.objects.create(
            conversation=conversation,
            role=Message.Role.ASSISTANT,
            content=welcome_text,
        )
        return (
            ChatResult(
                conversation_id=str(conversation.id),
                assistant_message_id=str(assistant_msg.id),
                text=assistant_msg.content,
            ),
            conversation,
        )

    if assistant_count < 3:
        system_prompt = build_onboarding_prompt(friend)
        extra_user_prompt = onboarding_question(assistant_count)
    else:
        recent_user_messages = (
            Message.objects.filter(conversation=conversation, role=Message.Role.USER)
            .order_by("-created_at")
            .values_list("content", flat=True)[:5]
        )

        inferred_mode = infer_conversation_mode(
            llm=llm,
            recent_messages=list(recent_user_messages),
        )

        maybe_update_conversation_mode(
            conversation=conversation,
            inferred_mode=inferred_mode,
        )
        system_prompt = build_system_prompt(
            friend=friend,
            memories=memories,
            mode=ConversationMode(conversation.current_mode),  # 🔹 usa o modo
        )
        extra_user_prompt = None

    recent_messages = get_recent_messages(conversation, limit=18)

    llm_messages = []

    if extra_user_prompt:
        llm_messages.append(LLMMessage(role="user", content=extra_user_prompt))

    llm_messages.append(LLMMessage(role="system", content=system_prompt))

    llm_messages.extend(
        LLMMessage(role=m.role, content=m.content) for m in recent_messages
    )

    # 5. Extração de perfil (permanece igual)
    extracted = extract_user_profile_from_message(
        llm=llm,
        user_text=user_text,
    )

    if extracted:
        profile, _ = UserSpiritualProfile.objects.get_or_create(user=friend.owner)
        profile.extracted_profile = merge_extracted_profile(
            profile.extracted_profile or {},
            extracted,
        )
        profile.save(update_fields=["extracted_profile", "updated_at"])

    # 6. Resposta da LLM
    resp = llm.chat(
        messages=llm_messages,
        temperature=0.6,
        max_tokens=150,
    )

    assistant_msg = Message.objects.create(
        conversation=conversation,
        role=Message.Role.ASSISTANT,
        content=resp.text,
        metadata=resp.raw or {"source_message_id": str(user_msg.id)},
    )

    maybe_extract_and_store_memory(
        friend=friend,
        conversation=conversation,
        user_text=user_text,
        assistant_text=assistant_msg.content,
        llm=llm,
    )

    image_analysis = infer_image_analysis(llm=llm, transcribed_text=resp.text)

    media_url = maybe_generate_image(image_analysis=image_analysis)

    return (
        ChatResult(
            conversation_id=str(conversation.id),
            assistant_message_id=str(assistant_msg.id),
            text=assistant_msg.content,
            media_url=media_url,
        ),
        conversation,
    )


def extract_user_profile_from_message(
    llm: LLMClient,
    user_text: str,
) -> dict:
    messages = [
        LLMMessage(role="system", content=build_profile_extraction_prompt()),
        LLMMessage(role="user", content=user_text),
    ]

    resp = llm.chat(
        messages=messages,
        temperature=0.0,  # determinístico
        max_tokens=300,
    )

    try:
        data = json.loads(resp.text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    return {}


def merge_extracted_profile(
    current: dict,
    extracted: dict,
) -> dict:
    merged = current.copy()

    for key, new_value in extracted.items():
        if key not in merged:
            merged[key] = new_value
            continue

        old_value = merged[key]

        # List merge (union, no duplicates)
        if isinstance(old_value, list) and isinstance(new_value, list):
            merged[key] = list(dict.fromkeys(old_value + new_value))
            continue

        # Dict merge (recursive)
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            merged[key] = merge_extracted_profile(old_value, new_value)
            continue

        # Scalar overwrite (string, int, bool)
        merged[key] = new_value

    return merged


def infer_conversation_mode(
    llm: LLMClient,
    recent_messages: list[str],
) -> Optional[ConversationMode]:
    messages = [
        LLMMessage(role="system", content=build_mode_inference_prompt()),
        LLMMessage(role="user", content="\n".join(recent_messages)),
    ]

    resp = llm.chat(
        messages=messages,
        temperature=0.0,
        max_tokens=120,
    )

    try:
        data = json.loads(resp.text)
        mode = data.get("conversation_mode")
        if mode and mode in ConversationMode._value2member_map_:
            return ConversationMode(mode)
    except Exception:
        pass

    return None


def maybe_update_conversation_mode(
    conversation: Conversation,
    inferred_mode: Optional[ConversationMode],
):
    if not inferred_mode:
        return

    current = ConversationMode(conversation.current_mode)

    if MODE_PRIORITY[inferred_mode] > MODE_PRIORITY[current]:
        conversation.current_mode = inferred_mode.value
        conversation.save(update_fields=["current_mode"])


def maybe_extract_and_store_memory(
    friend: VirtualFriend,
    conversation: Conversation,
    user_text: str,
    assistant_text: str,
    llm: LLMClient,
):
    result = extract_memory_with_llm(
        llm=llm,
        user_text=user_text,
        assistant_text=assistant_text,
        mode=conversation.current_mode,
    )

    if not result.get("should_create"):
        return None

    return upsert_memory(
        friend=friend,
        kind=result["kind"],
        key=result["key"],
        value=result["value"],
        confidence=result.get("confidence", 0.80),
        source={
            "conversation_id": str(conversation.id),
            "reason": result.get("reason"),
            "mode": conversation.current_mode,
        },
    )


def extract_memory_with_llm(
    llm: LLMClient,
    user_text: str,
    assistant_text: str,
    mode: str,
) -> dict:
    prompt = build_memory_prompt(
        user_text=user_text,
        assistant_text=assistant_text,
        mode=mode,
    )

    messages = [
        LLMMessage(role="system", content=prompt),
    ]

    resp = llm.chat(
        messages=messages,
        temperature=0.0,  # memória precisa ser determinística
        max_tokens=300,
    )

    try:
        data = json.loads(resp.text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    # fallback seguro
    return {"should_create": False}


def infer_image_analysis(
    llm: LLMClient,
    transcribed_text: str,
) -> Optional[ImageAnalysis]:
    """
    Analyzes a transcribed text and determines whether a contemplative image
    should be generated, returning structured visual parameters if applicable.
    """

    messages = [
        LLMMessage(
            role="system",
            content=IMAGE_EXTRACTION_PROMPT,
        ),
        LLMMessage(
            role="user",
            content=transcribed_text,
        ),
    ]

    resp = llm.chat(
        messages=messages,
        temperature=0.0,
        max_tokens=220,
    )

    try:
        data = json.loads(resp.text)

        if data.get("should_generate_image"):
            return ImageAnalysis(
                should_generate_image=True,
                image_type=data.get("image_type"),
                visual_elements=data.get("visual_elements", []),
                emotional_tone=data.get("emotional_tone"),
                visual_description=data.get("visual_description"),
            )

    except Exception as ex:
        logger.error(f"Failed to infer image analysis: {ex}")

    return ImageAnalysis()
