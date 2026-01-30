from enum import Enum

from django.db import models


class Gender(models.TextChoices):
    MALE = "male", "Male"
    FEMALE = "female", "Female"
    UNKNOWN = "unknown", "Unknown"


biblical_names = [
    {"name": "Adão", "gender": Gender.MALE},
    {"name": "Noé", "gender": Gender.MALE},
    {"name": "Abraão", "gender": Gender.MALE},
    {"name": "Isaque", "gender": Gender.MALE},
    {"name": "Rebeca", "gender": Gender.FEMALE},
    {"name": "Jacó", "gender": Gender.MALE},
    {"name": "Raquel", "gender": Gender.FEMALE},
    {"name": "José", "gender": Gender.MALE},
    {"name": "Moisés", "gender": Gender.MALE},
    {"name": "Arão", "gender": Gender.MALE},
    {"name": "Josué", "gender": Gender.MALE},
    {"name": "Calebe", "gender": Gender.MALE},
    {"name": "Ester", "gender": Gender.FEMALE},
    {"name": "Davi", "gender": Gender.MALE},
    {"name": "Salomão", "gender": Gender.MALE},
    {"name": "Elias", "gender": Gender.MALE},
    {"name": "Eliseu", "gender": Gender.MALE},
    {"name": "Isaías", "gender": Gender.MALE},
    {"name": "Jeremias", "gender": Gender.MALE},
    {"name": "Daniel", "gender": Gender.MALE},
    {"name": "Jó", "gender": Gender.MALE},
    {"name": "Jonas", "gender": Gender.MALE},
    {"name": "João Batista", "gender": Gender.MALE},
    {"name": "Pedro", "gender": Gender.MALE},
    {"name": "André", "gender": Gender.MALE},
    {"name": "Tiago", "gender": Gender.MALE},
    {"name": "João", "gender": Gender.MALE},
    {"name": "Filipe", "gender": Gender.MALE},
    {"name": "Bartolomeu", "gender": Gender.MALE},
    {"name": "Mateus", "gender": Gender.MALE},
    {"name": "Tomé", "gender": Gender.MALE},
    {"name": "Paulo", "gender": Gender.MALE},
    {"name": "Timóteo", "gender": Gender.MALE},
    {"name": "Lucas", "gender": Gender.MALE},
    {"name": "Marcos", "gender": Gender.MALE},
    {"name": "Áquila", "gender": Gender.MALE},
    {"name": "Barnabé", "gender": Gender.MALE},
]


class ConversationMode(str, Enum):
    LISTENING = "listening"  # escuta humana
    REFLECTIVE = "reflective"  # espiritual leve
    SPIRITUAL_AWARENESS = "spiritual_awareness"
    BIBLICAL = "biblical"  # bíblico


MODE_PRIORITY = {
    ConversationMode.LISTENING: 1,
    ConversationMode.REFLECTIVE: 2,
    ConversationMode.SPIRITUAL_AWARENESS: 3,
    ConversationMode.BIBLICAL: 4,
}
