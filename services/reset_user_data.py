"""
Reset User Data Use Case.

This module provides functionality for permanently deleting all user data
from the system, including profile, messages, and related entities.
"""

import logging
from typing import Optional

from django.db import transaction

from core.models import Message, Profile

logger = logging.getLogger(__name__)


class ResetUserDataUseCase:
    """
    Use case for permanently deleting all user data from the system.

    This class handles the complete deletion of a user's data including:
    - User profile
    - All messages (conversations)
    - Any other user-related data

    The deletion is performed within a database transaction to ensure atomicity.
    """

    @staticmethod
    @transaction.atomic
    def execute(user_identifier: str) -> bool:
        """
        Permanently delete all data for a user.

        This method:
        1. Finds the user by their identifier (telegram_user_id)
        2. Deletes all related data in the proper order
        3. Deletes the user profile
        4. Uses a transaction to ensure atomicity

        Args:
            user_identifier: The unique identifier for the user
                           (e.g., telegram_user_id, phone_number)

        Returns:
            bool: True if data was deleted, False if user not found

        Raises:
            Exception: If deletion fails (transaction will be rolled back)
        """
        try:
            # Try to find the user profile
            profile = ResetUserDataUseCase._find_profile(user_identifier)

            if not profile:
                logger.info(
                    f"No profile found for identifier: {user_identifier}. "
                    "Nothing to delete."
                )
                return False

            profile_id = profile.id
            profile_name = profile.name

            logger.info(
                f"Starting data deletion for profile {profile_id} ({profile_name})"
            )

            # Delete all messages (CASCADE should handle this, but we'll be explicit)
            message_count = Message.objects.filter(profile=profile).count()
            Message.objects.filter(profile=profile).delete()
            logger.info(f"Deleted {message_count} messages for profile {profile_id}")

            # Delete the profile itself
            profile.delete()
            logger.info(
                f"Successfully deleted profile {profile_id} and all related data"
            )

            return True

        except Exception as e:
            logger.error(
                f"Error deleting user data for {user_identifier}: {str(e)}",
                exc_info=True,
            )
            raise

    @staticmethod
    def _find_profile(user_identifier: str) -> Optional[Profile]:
        """
        Find a user profile by identifier.

        Tries to find the profile by telegram_user_id first, then by phone_number.

        Args:
            user_identifier: The unique identifier for the user

        Returns:
            Optional[Profile]: The profile if found, None otherwise
        """
        # Try telegram_user_id first
        try:
            return Profile.objects.get(telegram_user_id=user_identifier)
        except Profile.DoesNotExist:
            pass

        # Try phone_number as fallback
        try:
            return Profile.objects.get(phone_number=user_identifier)
        except Profile.DoesNotExist:
            pass

        return None
