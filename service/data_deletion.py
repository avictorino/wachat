"""
Data deletion service for handling user data deletion requests.

This module provides functionality to safely delete user data in compliance
with Meta's data deletion requirements.
"""

import logging
import re
from functools import wraps
from typing import Optional

from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import transaction

logger = logging.getLogger(__name__)


def normalize_phone_number(phone: str) -> str:
    """
    Normalize phone number by removing spaces, dashes, and parentheses.

    Args:
        phone: Phone number string potentially containing formatting characters

    Returns:
        Normalized phone number string with only digits and + prefix
    """
    # Remove spaces, dashes, and parentheses
    normalized = re.sub(r"[\s\-\(\)]", "", phone)
    return normalized


def mask_phone_number(phone: str) -> str:
    """
    Mask phone number for safe logging.
    Shows only first 3 and last 2 digits.

    Args:
        phone: Phone number to mask

    Returns:
        Masked phone number string
    """
    if len(phone) <= 5:
        return "***"
    return f"{phone[:3]}...{phone[-2:]}"


def rate_limit_by_ip(max_requests: int = 5, window_seconds: int = 3600):
    """
    Decorator to rate limit requests by IP address.

    Args:
        max_requests: Maximum number of requests allowed in the time window
        window_seconds: Time window in seconds (default: 1 hour)

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(request, *args, **kwargs):
            # Get client IP address
            x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
            if x_forwarded_for:
                ip = x_forwarded_for.split(",")[0].strip()
            else:
                ip = request.META.get("REMOTE_ADDR")

            # Create cache key
            cache_key = f"data_deletion_rate_limit_{ip}"

            # Get current request count
            request_count = cache.get(cache_key, 0)

            if request_count >= max_requests:
                # Rate limit exceeded
                return None  # Caller should handle None response

            # Increment request count
            cache.set(cache_key, request_count + 1, window_seconds)

            return func(request, *args, **kwargs)

        return wrapper

    return decorator


@transaction.atomic
def delete_user_data(phone_number: str) -> tuple[bool, Optional[str]]:
    """
    Delete user and all related data by phone number.

    This function performs the deletion within a transaction to ensure
    atomicity. All related data is automatically deleted via CASCADE.

    Args:
        phone_number: Normalized phone number to delete

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    try:
        # Normalize phone number
        normalized_phone = normalize_phone_number(phone_number)

        # Mask phone number for logging
        masked_phone = mask_phone_number(normalized_phone)

        # Find user by username (which stores the phone number)
        user = User.objects.filter(username=normalized_phone).first()

        if user:
            # Log deletion request (with masked phone)
            logger.info(
                f"Deleting user data for phone: {masked_phone}, " f"user_id: {user.id}"
            )

            # Delete user (CASCADE will handle all related data)
            user.delete()

            logger.info(f"Successfully deleted user data for phone: {masked_phone}")
            return True, None
        else:
            # User not found - but don't reveal this in logs
            logger.info(f"Data deletion request for phone: {masked_phone}")
            return True, None  # Return success even if user doesn't exist

    except Exception as e:
        masked_phone = mask_phone_number(phone_number)
        logger.error(
            f"Error deleting user data for phone: {masked_phone}, error: {str(e)}"
        )
        return False, "An error occurred processing your request."
