"""
Example Usage of Input Sanitization Service

This file demonstrates how to use the input sanitization service
in different scenarios.
"""

from services.input_sanitizer import sanitize_input, get_sanitizer, InputSanitizer


def example_1_basic_usage():
    """Example 1: Basic usage with the convenience function."""
    print("=== Example 1: Basic Usage ===\n")
    
    # Clean input - passes through unchanged
    clean_text = "Olá, preciso de ajuda espiritual"
    result = sanitize_input(clean_text)
    print(f"Input:  {clean_text}")
    print(f"Output: {result}")
    print(f"Changed: {clean_text != result}\n")
    
    # Harmful input - gets sanitized
    harmful_text = "Quero falar sobre sexo e morte"
    result = sanitize_input(harmful_text)
    print(f"Input:  {harmful_text}")
    print(f"Output: {result}")
    print(f"Changed: {harmful_text != result}\n")


def example_2_singleton_pattern():
    """Example 2: Using the singleton pattern."""
    print("=== Example 2: Singleton Pattern ===\n")
    
    # Get the singleton instance
    sanitizer = get_sanitizer()
    
    # Use it multiple times
    texts = [
        "Estou me sentindo sozinho",
        "Penso em suicídio às vezes",
        "O que você acha sobre aborto?",
    ]
    
    for text in texts:
        result = sanitizer.sanitize(text)
        print(f"Input:  {text}")
        print(f"Output: {result}\n")


def example_3_logging_control():
    """Example 3: Controlling logging behavior."""
    print("=== Example 3: Logging Control ===\n")
    
    text = "Quero conversar sobre pornografia"
    
    # With logging (default)
    print("With logging enabled:")
    result1 = sanitize_input(text, log_detections=True)
    print(f"Result: {result1}\n")
    
    # Without logging
    print("With logging disabled:")
    result2 = sanitize_input(text, log_detections=False)
    print(f"Result: {result2}\n")


def example_4_before_llm_call():
    """Example 4: Typical usage before an LLM call."""
    print("=== Example 4: Before LLM Call ===\n")
    
    # Simulating user input from a chat interface
    user_name = "João da Silva"
    user_message = "Estou com dúvidas sobre sexualidade"
    
    # Sanitize both before sending to LLM
    safe_name = sanitize_input(user_name)
    safe_message = sanitize_input(user_message)
    
    print(f"Original name:    {user_name}")
    print(f"Sanitized name:   {safe_name}")
    print(f"Original message: {user_message}")
    print(f"Sanitized message: {safe_message}\n")
    
    # Now safe_name and safe_message can be used in LLM prompts
    prompt = f"Usuário {safe_name} disse: {safe_message}"
    print(f"LLM prompt: {prompt}\n")


def example_5_edge_cases():
    """Example 5: Handling edge cases."""
    print("=== Example 5: Edge Cases ===\n")
    
    test_cases = [
        None,           # None input
        "",             # Empty string
        "   ",          # Whitespace only
        123,            # Non-string input
    ]
    
    for test in test_cases:
        result = sanitize_input(test)
        print(f"Input:  {repr(test)}")
        print(f"Output: {repr(result)}\n")


def example_6_multiple_categories():
    """Example 6: Text with multiple harmful categories."""
    print("=== Example 6: Multiple Categories ===\n")
    
    # Text containing multiple types of harmful content
    text = "Quero falar sobre sexo, morte e terrorismo"
    result = sanitize_input(text)
    
    print(f"Input:  {text}")
    print(f"Output: {result}")
    print("Note: Multiple harmful terms from different categories are sanitized\n")


def example_7_preserving_context():
    """Example 7: Preserving conversational context."""
    print("=== Example 7: Preserving Context ===\n")
    
    # The sanitizer preserves the flow of conversation
    text = "Olá, estou passando por um momento difícil. Penso muito em morte e tenho dúvidas sobre sexualidade. Pode me ajudar?"
    result = sanitize_input(text)
    
    print(f"Input:\n{text}\n")
    print(f"Output:\n{result}\n")
    print("Note: Harmful terms are replaced but the message structure is preserved\n")


def example_8_portuguese_variations():
    """Example 8: Portuguese language variations."""
    print("=== Example 8: Portuguese Variations ===\n")
    
    variations = [
        "sexo",       # noun
        "sexual",     # adjective
        "sexuais",    # plural adjective
        "morrer",     # infinitive verb
        "morrendo",   # gerund
        "morreu",     # past tense
    ]
    
    for word in variations:
        result = sanitize_input(word)
        print(f"{word:15} -> {result}")


if __name__ == "__main__":
    """Run all examples."""
    print("\n" + "="*60)
    print("INPUT SANITIZATION SERVICE - USAGE EXAMPLES")
    print("="*60 + "\n")
    
    example_1_basic_usage()
    example_2_singleton_pattern()
    example_3_logging_control()
    example_4_before_llm_call()
    example_5_edge_cases()
    example_6_multiple_categories()
    example_7_preserving_context()
    example_8_portuguese_variations()
    
    print("="*60)
    print("All examples completed!")
    print("="*60 + "\n")
