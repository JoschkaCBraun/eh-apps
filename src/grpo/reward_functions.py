"""
Reward Functions for GRPO Training

Exact reward functions from the Unsloth notebook for format checking
and answer validation.
"""

from typing import List, Dict, Any, Optional
from .formatting_utils import FormatChecker


# Global variables for printing (from notebook)
PRINTED_TIMES = 0
PRINT_EVERY_STEPS = 5


def match_format_exactly(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function to match format exactly - rewards with 3 points if it succeeds.
    
    Args:
        completions: List of completion lists (each completion is a list with one dict)
        **kwargs: Additional arguments (unused)
        
    Returns:
        List of scores (3.0 if format matches, 0.0 otherwise)
    """
    # Initialize format checker
    tokenizer = kwargs.get('tokenizer', None)
    checker = FormatChecker(tokenizer)
    
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if checker.check_format_exact(response):
            score += 3.0
        scores.append(score)
    return scores


def match_format_approximately(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Reward function if format partially matches by counting each symbol.
    
    Args:
        completions: List of completion lists
        **kwargs: Additional arguments (unused)
        
    Returns:
        List of scores based on tag presence
    """
    # Initialize format checker
    tokenizer = kwargs.get('tokenizer', None)
    checker = FormatChecker(tokenizer)
    
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        
        # No need to reward <start_working_out> since we always prepend it!
        # score += 0.5 if response.count(checker.reasoning_start) == 1 else -1.0
        score += 0.5 if response.count(checker.reasoning_end) == 1 else -1.0
        score += 0.5 if response.count(checker.solution_start) == 1 else -1.0
        score += 0.5 if response.count(checker.solution_end) == 1 else -1.0
        
        scores.append(score)
    return scores


def check_answer(prompts: List[List[Dict]], completions: List[List[Dict]], 
                answer: List[str], **kwargs) -> List[float]:
    """
    Extract generated answer and reward/penalize based on correctness.
    Also rewards based on how close the answer is to the true one via ratios.
    
    Args:
        prompts: List of prompt message lists
        completions: List of completion lists
        answer: List of true answers
        **kwargs: Additional arguments
        
    Returns:
        List of scores based on answer correctness
    """
    # Get the question from the last user message
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
    # Initialize format checker
    tokenizer = kwargs.get('tokenizer', None)
    checker = FormatChecker(tokenizer)
    
    # Extract answers using the format pattern
    extracted_responses = []
    for r in responses:
        extracted = checker.extract_answer(r)
        extracted_responses.append(extracted)
    
    scores = []
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        if guess is None:
            scores.append(-2.0)
            continue
        
        # Correct answer gets 5 points!
        if guess == true_answer:
            score += 5.0
        # Match if spaces are seen, but less reward
        elif guess.strip() == true_answer.strip():
            score += 3.5
        else:
            # We also reward it if the answer is close via ratios!
            # Ie if the answer is within some range, reward it!
            try:
                ratio = float(guess) / float(true_answer)
                if ratio >= 0.9 and ratio <= 1.1:
                    score += 2.0
                elif ratio >= 0.8 and ratio <= 1.2:
                    score += 1.5
                else:
                    score -= 2.5  # Penalize wrong answers
            except:
                score -= 4.5  # Penalize non-numeric or invalid answers
        
        scores.append(score)
    return scores


def check_numbers(prompts: List[List[Dict]], completions: List[List[Dict]], 
                 answer: List[str], **kwargs) -> List[float]:
    """
    Check numeric answers with float conversion and print progress.
    
    Args:
        prompts: List of prompt message lists
        completions: List of completion lists
        answer: List of true answers
        **kwargs: Additional arguments
        
    Returns:
        List of scores based on numeric answer correctness
    """
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]
    
    # Initialize format checker
    tokenizer = kwargs.get('tokenizer', None)
    checker = FormatChecker(tokenizer)
    
    # Extract numbers using the number pattern
    extracted_responses = []
    for r in responses:
        extracted = checker.extract_number(r)
        extracted_responses.append(extracted)
    
    scores = []
    
    # Print only every few steps
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print(
            '*' * 20 + f"Question:\n{question}", 
            f"\nAnswer:\n{answer[0]}", 
            f"\nResponse:\n{responses[0]}", 
            f"\nExtracted:\n{extracted_responses[0]}"
        )
    PRINTED_TIMES += 1
    
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(-2.5)
            continue
        
        # Convert to numbers
        try:
            true_answer = float(true_answer.strip())
            # Remove commas like in 123,456
            guess = float(guess.strip().replace(",", ""))
            scores.append(3.5 if guess == true_answer else -1.5)
        except:
            scores.append(0)
            continue
    
    return scores


def create_reward_functions(tokenizer=None) -> List:
    """
    Create the list of reward functions for GRPO training.
    
    Args:
        tokenizer: Optional tokenizer for format checking
        
    Returns:
        List of reward functions in the order used by the notebook
    """
    # Create wrapper functions that include the tokenizer
    def _match_format_exactly(completions, **kwargs):
        return match_format_exactly(completions, tokenizer=tokenizer, **kwargs)
    
    def _match_format_approximately(completions, **kwargs):
        return match_format_approximately(completions, tokenizer=tokenizer, **kwargs)
    
    def _check_answer(prompts, completions, answer, **kwargs):
        return check_answer(prompts, completions, answer, tokenizer=tokenizer, **kwargs)
    
    def _check_numbers(prompts, completions, answer, **kwargs):
        return check_numbers(prompts, completions, answer, tokenizer=tokenizer, **kwargs)
    
    return [
        _match_format_exactly,
        _match_format_approximately,
        _check_answer,
        _check_numbers,
    ]


def reset_print_counter():
    """Reset the global print counter for new training runs."""
    global PRINTED_TIMES
    PRINTED_TIMES = 0


def test_reward_functions():
    """Test reward functions with sample data."""
    print("Testing Reward Functions")
    print("=" * 80)
    
    # Sample completions
    good_completion = [{
        "content": "Let me think<end_working_out><SOLUTION>42</SOLUTION>"
    }]
    
    bad_completion = [{
        "content": "The answer is 42"
    }]
    
    partial_completion = [{
        "content": "<end_working_out>The answer is <SOLUTION>42"
    }]
    
    # Test format exact matching
    print("\nTesting match_format_exactly:")
    print(f"  Good: {match_format_exactly([good_completion])}")
    print(f"  Bad: {match_format_exactly([bad_completion])}")
    print(f"  Partial: {match_format_exactly([partial_completion])}")
    
    # Test format approximate matching
    print("\nTesting match_format_approximately:")
    print(f"  Good: {match_format_approximately([good_completion])}")
    print(f"  Bad: {match_format_approximately([bad_completion])}")
    print(f"  Partial: {match_format_approximately([partial_completion])}")
    
    # Test answer checking
    print("\nTesting check_answer:")
    prompts = [[{"role": "user", "content": "What is 6*7?"}]]
    print(f"  Correct: {check_answer(prompts, [good_completion], ['42'])}")
    print(f"  Wrong: {check_answer(prompts, [good_completion], ['43'])}")
    print(f"  Close: {check_answer(prompts, [good_completion], ['41'])}")  # Within 10%
    
    # Test number checking
    print("\nTesting check_numbers:")
    number_completion = [{
        "content": "<SOLUTION>123,456.78</SOLUTION>"
    }]
    print(f"  Number extraction: {check_numbers(prompts, [number_completion], ['123456.78'])}")


if __name__ == "__main__":
    test_reward_functions()