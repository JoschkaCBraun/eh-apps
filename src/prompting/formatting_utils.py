"""
Formatting Utilities for GRPO Training

Implements regex patterns and extraction utilities matching the notebook's
exact format checking and answer extraction logic.
"""

import re
from typing import Optional, Pattern


class FormatChecker:
    """Check and extract formatted responses from model outputs."""
    
    def __init__(self, tokenizer=None):
        """
        Initialize format checker with tag patterns.
        
        Args:
            tokenizer: Optional tokenizer for EOS token handling
        """
        # Define the exact tags from notebook
        self.reasoning_start = "<start_working_out>"
        self.reasoning_end = "<end_working_out>"
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"
        
        # Store tokenizer for EOS token
        self.tokenizer = tokenizer
        self.eos_token = tokenizer.eos_token if tokenizer else ""
        
        # Create regex patterns
        self._create_patterns()
    
    def _create_patterns(self):
        """Create regex patterns for format matching."""
        # Pattern for exact format matching (from notebook)
        # Add optional EOS token matching
        solution_end_regex = r"</SOLUTION>[\s]{0,}"
        if self.eos_token:
            solution_end_regex += "(?:" + re.escape(self.eos_token) + ")?"
        
        # Main format pattern
        self.match_format = re.compile(
            rf"{re.escape(self.reasoning_end)}.*?"
            rf"{re.escape(self.solution_start)}(.+?){solution_end_regex}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )
        
        # Number extraction pattern (from notebook)
        self.match_numbers = re.compile(
            re.escape(self.solution_start) + r".*?[\s]{0,}([-]?[\d\.\,]{1,})",
            flags=re.MULTILINE | re.DOTALL
        )
        
        # Pattern to check individual tags
        self.tag_patterns = {
            'reasoning_start': re.compile(re.escape(self.reasoning_start)),
            'reasoning_end': re.compile(re.escape(self.reasoning_end)),
            'solution_start': re.compile(re.escape(self.solution_start)),
            'solution_end': re.compile(re.escape(self.solution_end))
        }
    
    def check_format_exact(self, text: str) -> bool:
        """
        Check if text matches the exact expected format.
        
        Args:
            text: Generated text to check
            
        Returns:
            True if format matches exactly
        """
        return self.match_format.search(text) is not None
    
    def extract_answer(self, text: str) -> Optional[str]:
        """
        Extract the answer from formatted text.
        
        Args:
            text: Generated text with format tags
            
        Returns:
            Extracted answer or None if not found
        """
        match = self.match_format.search(text)
        if match:
            return match.group(1)
        return None
    
    def extract_number(self, text: str) -> Optional[str]:
        """
        Extract a number from the solution section.
        
        Args:
            text: Generated text with solution tags
            
        Returns:
            Extracted number string or None if not found
        """
        match = self.match_numbers.search(text)
        if match:
            return match.group(1)
        return None
    
    def count_tags(self, text: str) -> dict:
        """
        Count occurrences of each format tag.
        
        Args:
            text: Generated text to analyze
            
        Returns:
            Dictionary with tag counts
        """
        counts = {}
        counts['reasoning_start'] = text.count(self.reasoning_start)
        counts['reasoning_end'] = text.count(self.reasoning_end)
        counts['solution_start'] = text.count(self.solution_start)
        counts['solution_end'] = text.count(self.solution_end)
        return counts
    
    def validate_format(self, text: str) -> dict:
        """
        Validate format and return detailed analysis.
        
        Args:
            text: Generated text to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'is_valid': False,
            'has_exact_format': False,
            'tag_counts': self.count_tags(text),
            'extracted_answer': None,
            'extracted_number': None,
            'issues': []
        }
        
        # Check tag counts
        counts = result['tag_counts']
        
        # Check for missing or duplicate tags
        # Note: reasoning_start is prepended, so we don't check it
        if counts['reasoning_end'] != 1:
            result['issues'].append(f"reasoning_end count: {counts['reasoning_end']} (expected 1)")
        if counts['solution_start'] != 1:
            result['issues'].append(f"solution_start count: {counts['solution_start']} (expected 1)")
        if counts['solution_end'] != 1:
            result['issues'].append(f"solution_end count: {counts['solution_end']} (expected 1)")
        
        # Check exact format
        result['has_exact_format'] = self.check_format_exact(text)
        
        # Extract answer if possible
        result['extracted_answer'] = self.extract_answer(text)
        result['extracted_number'] = self.extract_number(text)
        
        # Determine overall validity
        result['is_valid'] = (
            result['has_exact_format'] and 
            len(result['issues']) == 0
        )
        
        return result


# Standalone utility functions matching notebook
def extract_answer(text: str, tokenizer=None) -> Optional[str]:
    """
    Extract answer from formatted text.
    
    Args:
        text: Generated text with format tags
        tokenizer: Optional tokenizer for EOS token
        
    Returns:
        Extracted answer or None
    """
    checker = FormatChecker(tokenizer)
    return checker.extract_answer(text)


def extract_number(text: str) -> Optional[str]:
    """
    Extract number from solution section.
    
    Args:
        text: Generated text with solution tags
        
    Returns:
        Extracted number string or None
    """
    checker = FormatChecker()
    return checker.extract_number(text)


def test_format_patterns():
    """Test the format patterns with examples from notebook."""
    print("Testing Format Patterns")
    print("=" * 80)
    
    checker = FormatChecker()
    
    # Test cases from notebook
    test_cases = [
        ("Let me think!<end_working_out><SOLUTION>\n2\n</SOLUTION>", True, "\\n2\\n"),
        ("<start_working_out>Let me think!<end_working_out><SOLUTION>  2  </SOLUTION>\n\n", True, "  2  "),
        ("<SOLUTION>  0.34  </SOLUTION>", False, None),  # Missing reasoning_end
        ("<end_working_out><SOLUTION>  123,456  </SOLUTION>", True, "  123,456  "),
    ]
    
    for text, should_match, expected_answer in test_cases:
        print(f"\nTest: {text[:50]}...")
        result = checker.validate_format(text)
        print(f"  Has exact format: {result['has_exact_format']} (expected: {should_match})")
        print(f"  Extracted answer: {result['extracted_answer']} (expected: {expected_answer})")
        print(f"  Extracted number: {result['extracted_number']}")
        if result['issues']:
            print(f"  Issues: {result['issues']}")
    
    # Test number extraction
    print("\n" + "=" * 80)
    print("Testing Number Extraction")
    print("=" * 80)
    
    number_tests = [
        "<SOLUTION>  0.34  </SOLUTION>",
        "<SOLUTION>  123,456  </SOLUTION>",
        "<SOLUTION>  -0.234  </SOLUTION>",
        "<SOLUTION>17</SOLUTION>",
    ]
    
    for text in number_tests:
        number = checker.extract_number(text)
        print(f"{text} -> {number}")


if __name__ == "__main__":
    test_format_patterns()