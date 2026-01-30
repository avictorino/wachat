from abc import ABC, abstractmethod
from messaging.types import OutgoingMessage


class MessagingProvider(ABC):
    @abstractmethod
    def send(self, message: OutgoingMessage) -> None:
        pass