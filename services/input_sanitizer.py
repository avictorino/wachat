"""
Input Sanitization Service for LLM interactions.

This module provides a reusable safety layer that sanitizes user input
before it is sent to the LLM. It removes or neutralizes mentions of
harmful or disallowed topics to ensure safe conversations.

Key features:
- Language-aware filtering (Brazilian Portuguese)
- Flexible keyword matching with variations and slang
- Non-intrusive sanitization (no exceptions, no user-facing errors)
- Optional logging for monitoring
- Extendable design for future categories
"""

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class InputSanitizer:
    """
    Sanitizes user input by detecting and neutralizing harmful content.
    
    This class is designed as a protective layer that should be applied
    before every LLM call. It does not generate responses or call the LLM,
    it only prepares safe input.
    """

    def __init__(self):
        """Initialize the sanitizer with keyword patterns."""
        # Sexual content keywords (Portuguese)
        self.sexual_keywords = self._compile_patterns([
            r'\bsexo\b', r'\bsexual\b', r'\bsexuais\b', r'\bsexualidade\b',
            r'\btransar\b', r'\btransando\b', r'\btrepar\b',
            r'\bfoder\b', r'\bfodendo\b', r'\bfodeu\b',
            r'\bporno\b', r'\bpornografia\b', r'\bpornográfico\b',
            r'\bmasturbação\b', r'\bmasturbar\b',
            r'\borgasmo\b', r'\bejaculação\b',
            r'\bviagra\b', r'\berotic[oa]\b',
            r'\bpênis\b', r'\bvagina\b', r'\bseio\b', r'\bseios\b',
            r'\bpeito\b', r'\bpeitos\b',
            r'\bbunda\b', r'\brabo\b', r'\bcu\b',
            r'\bestupro\b', r'\bestupr[oa]d[oa]\b',
            r'\babuso\b.*\bsexual\b', r'\bsexual\b.*\babuso\b',
            r'\bincesto\b', r'\bpedofilia\b',
            r'\bnudes?\b', r'\bnua?\b', r'\bnus?\b',
            r'\bintim[oa]s?\b', r'\bíntim[oa]s?\b',
            r'\brelação\b.*\bsexual\b', r'\bsexual\b.*\brelação\b',
        ])

        # Death-related keywords (Portuguese)
        self.death_keywords = self._compile_patterns([
            r'\bmorte\b', r'\bmorrer\b', r'\bmorrendo\b', r'\bmorreu\b',
            r'\bfalecimento\b', r'\bfalecer\b', r'\bfaleceu\b',
            r'\bóbito\b', r'\bsuicídio\b', r'\bsuicid[ao]s?\b',
            r'\bsuicidar\b', r'\bsuicidas\b', r'\bmatar\b', r'\bmatar-se\b',
            r'\bmatando\b', r'\bmatou\b', r'\bassassin[oa]\b',
            r'\bhomicídio\b', r'\bfuner[aá]rio\b', r'\bfuneral\b',
            r'\bcaixão\b', r'\bentierro\b', r'\benterrar\b',
            r'\bcadáver\b', r'\bcorpo\b.*\bmorto\b',
            r'\bvida\b.*\btirar\b', r'\btirar\b.*\bvida\b',
            r'\benforcamento\b', r'\benforcar\b',
            r'\bveneno\b', r'\benvenenar\b',
            r'\bmortal\b', r'\bletal\b', r'\bfatal\b',
        ])

        # Controversial/polarizing topics keywords (Portuguese)
        self.controversial_keywords = self._compile_patterns([
            r'\baborto\b', r'\binterrupção\b.*\bgravidez\b',
            r'\babortar\b', r'\babortivo\b',
            r'\beutanásia\b', r'\bsuicídio\b.*\bassistido\b',
            r'\bgenocídio\b', r'\bextermínio\b',
            r'\bditadura\b', r'\bditador\b', r'\bautoritarismo\b',
            r'\bfascismo\b', r'\bfascista\b', r'\bnazismo\b', r'\bnazista\b',
            r'\bcomunismo\b', r'\bcomunista\b', r'\bsocialismo\b',
            r'\bterrorismo\b', r'\bterrorista\b',
            r'\bextremismo\b', r'\bextremista\b',
            r'\bradicalismo\b', r'\bradical\b.*\breligios[oa]\b',
            r'\bguerra\b.*\breligios[oa]\b', r'\breligios[oa]\b.*\bguerra\b',
            r'\bodio\b', r'\bpreconceito\b', r'\bdiscriminação\b',
            r'\bracismo\b', r'\bracista\b', r'\bxenofobia\b',
            r'\bhomofobia\b', r'\btransfobia\b',
            r'\bviolência\b.*\bpolític[oa]\b', r'\bpolític[oa]\b.*\bviolência\b',
        ])

        # Neutral replacement text
        self.neutral_placeholder = "[tema sensível]"

    def _compile_patterns(self, patterns: List[str]) -> List[re.Pattern]:
        """
        Compile regex patterns for efficient matching.
        
        Args:
            patterns: List of regex pattern strings
            
        Returns:
            List of compiled regex patterns
        """
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

    def _detect_harmful_content(self, text: str) -> Dict[str, List[str]]:
        """
        Detect harmful content categories in the text.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Dictionary mapping category names to lists of detected matches
        """
        detections = {
            'sexual': [],
            'death': [],
            'controversial': []
        }

        # Check for sexual content
        for pattern in self.sexual_keywords:
            matches = pattern.findall(text)
            if matches:
                detections['sexual'].extend(matches)

        # Check for death-related content
        for pattern in self.death_keywords:
            matches = pattern.findall(text)
            if matches:
                detections['death'].extend(matches)

        # Check for controversial content
        for pattern in self.controversial_keywords:
            matches = pattern.findall(text)
            if matches:
                detections['controversial'].extend(matches)

        return detections

    def _replace_harmful_terms(self, text: str, all_patterns: List[re.Pattern]) -> str:
        """
        Replace harmful terms with neutral placeholder.
        
        Args:
            text: The input text
            all_patterns: List of all compiled patterns to replace
            
        Returns:
            Text with harmful terms replaced
        """
        sanitized = text
        for pattern in all_patterns:
            sanitized = pattern.sub(self.neutral_placeholder, sanitized)
        
        # Clean up multiple consecutive placeholders
        sanitized = re.sub(
            rf'({re.escape(self.neutral_placeholder)}\s*){{2,}}',
            self.neutral_placeholder + ' ',
            sanitized
        )
        
        return sanitized.strip()

    def sanitize(self, text: str, log_detections: bool = True) -> str:
        """
        Sanitize user input by removing or neutralizing harmful content.
        
        This is the main public method that should be called before every
        LLM interaction. It:
        1. Detects harmful content in various categories
        2. Replaces harmful terms with neutral placeholders
        3. Optionally logs detections for monitoring
        4. Returns sanitized text safe for LLM processing
        
        Args:
            text: Raw user input to sanitize
            log_detections: Whether to log when sanitization occurs (default: True)
            
        Returns:
            Sanitized text safe for LLM processing
            
        Note:
            - This method never raises exceptions
            - If input is None or empty, returns empty string
            - Sanitization is transparent to the user
            - Original harmful content is never stored in logs
        """
        # Handle edge cases
        if not text:
            return ""
        
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}")
            return str(text)

        try:
            # Detect harmful content
            detections = self._detect_harmful_content(text)
            
            # Check if any harmful content was detected
            has_harmful_content = any(
                len(matches) > 0 for matches in detections.values()
            )
            
            if not has_harmful_content:
                # Input is clean, return as-is
                return text
            
            # Log detection (without storing the actual harmful content)
            if log_detections:
                detected_categories = [
                    category for category, matches in detections.items()
                    if len(matches) > 0
                ]
                logger.info(
                    f"Input sanitization performed. Categories detected: {detected_categories}",
                    extra={
                        'categories': detected_categories,
                        'input_length': len(text),
                    }
                )
            
            # Combine all patterns for replacement
            all_patterns = (
                self.sexual_keywords +
                self.death_keywords +
                self.controversial_keywords
            )
            
            # Replace harmful terms
            sanitized_text = self._replace_harmful_terms(text, all_patterns)
            
            return sanitized_text
            
        except Exception as e:
            # Never let sanitization errors break the application
            logger.error(
                f"Error during input sanitization: {str(e)}",
                exc_info=True
            )
            # In case of error, return original text to avoid breaking the flow
            # This is a safety fallback - in production, you might want to
            # be more conservative and return a safe default instead
            return text


# Global singleton instance for easy reuse
_sanitizer_instance = None


def get_sanitizer() -> InputSanitizer:
    """
    Get or create the global InputSanitizer instance.
    
    This function provides a singleton pattern for the sanitizer,
    ensuring consistent behavior across the application.
    
    Returns:
        The global InputSanitizer instance
        
    Example:
        >>> from services.input_sanitizer import get_sanitizer
        >>> sanitizer = get_sanitizer()
        >>> safe_input = sanitizer.sanitize(user_input)
    """
    global _sanitizer_instance
    if _sanitizer_instance is None:
        _sanitizer_instance = InputSanitizer()
    return _sanitizer_instance


def sanitize_input(text: str, log_detections: bool = True) -> str:
    """
    Convenience function to sanitize input using the global sanitizer.
    
    This is a shorthand for get_sanitizer().sanitize(text).
    
    Args:
        text: Raw user input to sanitize
        log_detections: Whether to log when sanitization occurs
        
    Returns:
        Sanitized text safe for LLM processing
        
    Example:
        >>> from services.input_sanitizer import sanitize_input
        >>> safe_input = sanitize_input(user_input)
    """
    return get_sanitizer().sanitize(text, log_detections=log_detections)
