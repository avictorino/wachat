#!/usr/bin/env python
"""
Manual validation script for addiction intent detection.

This script demonstrates that addiction-related intents are properly
detected and handled with empathy and without judgment.
"""

import os
import sys

# Add the project to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock environment for testing
os.environ.setdefault("GROQ_API_KEY", "test-key-for-validation")
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from unittest.mock import Mock, patch


def validate_addiction_intents():
    """Validate that addiction intents are properly configured."""
    print("=" * 70)
    print("ADDICTION INTENT VALIDATION")
    print("=" * 70)
    print()
    
    from services.groq_service import GroqService
    
    # Mock the Groq client
    with patch("services.groq_service.Groq") as mock_groq_client:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        
        mock_groq_instance = Mock()
        mock_groq_instance.chat.completions.create.return_value = mock_response
        mock_groq_client.return_value = mock_groq_instance
        
        service = GroqService()
        
        # Test 1: Verify addiction intents are in valid lists
        print("✓ Test 1: Checking valid intent lists...")
        addiction_intents = ["drogas", "alcool", "sexo", "cigarro"]
        
        for intent in addiction_intents:
            mock_response.choices[0].message.content = intent
            detected = service.detect_intent(f"Test message for {intent}")
            if detected == intent:
                print(f"  ✓ Intent '{intent}' is properly recognized")
            else:
                print(f"  ✗ FAILED: Intent '{intent}' was not recognized (got '{detected}')")
                return False
        
        print()
        
        # Test 2: Verify theme approximation works
        print("✓ Test 2: Checking theme approximation...")
        theme_mappings = {
            "dependência química": "drogas",
            "bebida": "alcool",
            "compulsão sexual": "sexo",
            "fumo": "cigarro",
        }
        
        for input_theme, expected_theme in theme_mappings.items():
            mock_response.choices[0].message.content = expected_theme
            approximated = service.approximate_theme(input_theme)
            if approximated == expected_theme:
                print(f"  ✓ '{input_theme}' → '{expected_theme}'")
            else:
                print(f"  ✗ FAILED: '{input_theme}' mapped to '{approximated}' instead of '{expected_theme}'")
                return False
        
        print()
        
        # Test 3: Verify response generation includes empathy guidance
        print("✓ Test 3: Checking response generation...")
        mock_response.choices[0].message.content = "Understanding response"
        
        for intent in addiction_intents:
            try:
                response = service.generate_intent_response(
                    user_message=f"I'm struggling with {intent}",
                    intent=intent,
                    name="Test User",
                    inferred_gender="unknown"
                )
                
                # Check that the system prompt contains empathy guidance
                call_args = mock_groq_instance.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                system_message = next(m for m in messages if m["role"] == "system")
                system_content = system_message["content"].lower()
                
                # Check for empathy keywords
                empathy_keywords = ["sem julgamento", "empático", "condição real", "não como fraqueza"]
                has_empathy = any(keyword in system_content for keyword in empathy_keywords)
                
                if has_empathy:
                    print(f"  ✓ Intent '{intent}' includes empathy guidance")
                else:
                    print(f"  ✗ FAILED: Intent '{intent}' missing empathy guidance")
                    print(f"    System prompt: {system_content[:200]}...")
                    return False
                    
            except Exception as e:
                print(f"  ✗ FAILED: Intent '{intent}' raised error: {e}")
                return False
        
        print()
        print("=" * 70)
        print("✓ ALL VALIDATION CHECKS PASSED!")
        print("=" * 70)
        print()
        print("Summary:")
        print("- Addiction intents (drogas, alcool, sexo, cigarro) are recognized")
        print("- Theme approximation maps variations to addiction intents")
        print("- Response generation includes empathetic, non-judgmental guidance")
        print("- Addiction is treated as a real condition, not moral failure")
        print()
        return True


def show_intent_guidance():
    """Display the guidance configured for each addiction intent."""
    print("=" * 70)
    print("ADDICTION INTENT GUIDANCE SUMMARY")
    print("=" * 70)
    print()
    
    guidance_summary = {
        "drogas": "Drug dependency - treated as real condition, not moral failure",
        "alcool": "Alcohol dependency - treated as real condition, not moral failure",
        "sexo": "Sexual compulsion - treated as real condition, not moral failure",
        "cigarro": "Smoking/nicotine dependency - treated as real condition, not moral failure",
    }
    
    for intent, description in guidance_summary.items():
        print(f"✓ {intent.upper()}")
        print(f"  {description}")
        print()
    
    print("Key principles for addiction intents:")
    print("- Empathetic and serious tone")
    print("- No judgment or moralization")
    print("- Normalize struggle without normalizing behavior")
    print("- No religious language as punishment")
    print("- No pressure for immediate abstinence")
    print("- Encourage reflection and seeking help")
    print()


if __name__ == "__main__":
    print()
    show_intent_guidance()
    
    if validate_addiction_intents():
        print("✓ Validation successful!")
        sys.exit(0)
    else:
        print("✗ Validation failed!")
        sys.exit(1)
