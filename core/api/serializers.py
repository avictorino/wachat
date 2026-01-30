from rest_framework import serializers
from biblical_friend.models import VirtualFriend, Conversation, Message


class VirtualFriendSerializer(serializers.ModelSerializer):
    class Meta:
        model = VirtualFriend
        fields = ["id", "name", "persona", "tone", "age", "background", "is_active", "created_at"]


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ["id", "role", "content", "metadata", "created_at"]


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)

    class Meta:
        model = Conversation
        fields = ["id", "title", "context", "is_closed", "created_at", "messages"]