"""
Answer Extractor for Model Evaluation

Extracts and validates answers from model outputs using tag-based parsing.
"""

import re
from typing import Dict, Optional


class AnswerExtractor:
    """Extracts answers from model outputs using tag-based parsing."""
    
    def __init__(self):
        self.thinking_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        # Patterns for counting for loops and function definitions
        self.for_loop_pattern = re.compile(r'\bfor\s+.*?:')
        self.function_def_pattern = re.compile(r'\bdef\s+\w+\s*\(')
        self.curly_brace_pattern = re.compile(r'[{}]')
        # Patterns for set() and dict() constructor calls
        self.set_constructor_pattern = re.compile(r'\bset\(')
        self.dict_constructor_pattern = re.compile(r'\bdict\(')
    
    def extract_answer(self, model_output: str) -> Dict:
        """Extract answer components from model output.
        
        Args:
            model_output: Raw output from the model
        
        Returns:
            Dictionary containing:
                - full_output: Complete model output (always stored)
                - thinking: Content from <think> tags (empty string if not found)
                - answer: Content from <answer> tags (empty string if not found)
                - thinking_found: Boolean indicating if valid thinking tags were found
                - code_found: Boolean indicating if valid code tags were found
                - for_loop_count: Number of for loops in the code
                - function_def_count: Number of function definitions in the code
                - curly_brace_count: Number of curly braces in the code
                - set_constructor_count: Number of set() constructor calls
                - dict_constructor_count: Number of dict() constructor calls
        """
        # Always store the full output
        result = {
            "full_output": model_output,
            "think": "",
            "answer": "",
            "think_found": False,
            "answer_found": False,
            "for_loop_count": 0,
            "function_def_count": 0,
            "curly_brace_count": 0,
            "set_constructor_count": 0,
            "dict_constructor_count": 0
        }
        
        # Try optimal case first - properly formatted tags
        thinking_match = self.thinking_pattern.search(model_output)
        if thinking_match:
            result["think"] = thinking_match.group(1).strip()
            result["think_found"] = True
        else:
            # Fallback: look for exactly 2 occurrences of "thinking" (case insensitive)
            thinking_positions = []
            for match in re.finditer(r'</?think>', model_output, re.IGNORECASE):
                thinking_positions.append((match.start(), match.end()))
            
            if len(thinking_positions) >= 2:
                # Use outermost occurrences (first and last)
                start_pos = thinking_positions[0][1]  # After first tag
                end_pos = thinking_positions[-1][0]   # Before last tag
                result["think"] = model_output[start_pos:end_pos].strip()
                result["think_found"] = True
        
        # Extract code content
        answer_match = self.answer_pattern.search(model_output)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            result["answer"] = self._clean_code(answer_content)
            result["answer_found"] = True
        else:
            # Fallback: look for exactly 2 occurrences of "code" (case insensitive)
            answer_positions = []
            for match in re.finditer(r'</?answer>', model_output, re.IGNORECASE):
                answer_positions.append((match.start(), match.end()))
            
            if len(answer_positions) >= 2:
                # Use outermost occurrences (first and last)
                start_pos = answer_positions[0][1]  # After first tag
                end_pos = answer_positions[-1][0]   # Before last tag
                answer_content = model_output[start_pos:end_pos].strip()
                result["answer"] = self._clean_code(answer_content)
                result["answer_found"] = True
        
        # Count for loops and function definitions if code was found
        if result["answer_found"] and result["answer"]:
            result["for_loop_count"] = len(self.for_loop_pattern.findall(result["answer"]))
            result["function_def_count"] = len(self.function_def_pattern.findall(result["answer"]))
            result["curly_brace_count"] = len(self.curly_brace_pattern.findall(result["answer"]))
            result["set_constructor_count"] = len(self.set_constructor_pattern.findall(result["answer"]))
            result["dict_constructor_count"] = len(self.dict_constructor_pattern.findall(result["answer"]))
        
        return result
    
    def _clean_code(self, code: str) -> str:
        """Clean extracted code to ensure it's valid Python.
        
        Args:
            code: Raw code string
            
        Returns:
            Cleaned code string starting from first def
        """
        # Remove any markdown code fence markers
        code = re.sub(r'^```\w*\n?', '', code, flags=re.MULTILINE)
        code = re.sub(r'\n?```$', '', code, flags=re.MULTILINE)
        
        # Find the first def statement
        def_match = re.search(r'^def\s+', code, re.MULTILINE)
        if def_match:
            # Extract from the first def onwards
            code = code[def_match.start():]
        
        return code.strip()
    
    def extract_batch_answers(self, model_outputs: list) -> list:
        """Extract answers from a batch of model outputs.
        
        Args:
            model_outputs: List of model output strings
        
        Returns:
            List of extracted answer dictionaries
        """
        return [self.extract_answer(output) for output in model_outputs]
    
    def validate_tags(self, model_output: str) -> Dict:
        """Validate tag presence and format in model output.
        
        Args:
            model_output: Raw output from the model
        
        Returns:
            Dictionary with validation results
        """
        # Check for opening tags
        think_open = '<think>' in model_output
        think_close = '</think>' in model_output
        answer_open = '<answer>' in model_output
        answer_close = '</answer>' in model_output
        
        # Check for proper tag pairs
        think_valid = think_open and think_close
        answer_valid = answer_open and answer_close
        
        # Check for tag order (thinking should come before code)
        think_before_answer = True
        if think_valid and answer_valid:
            think_start = model_output.find('<think>')
            answer_start = model_output.find('<answer>')
            think_before_answer = think_start < answer_start
        
        return {
            "think_tags_valid": think_valid,
            "answer_tags_valid": answer_valid,
            "think_before_answer": think_before_answer,
            "has_any_tags": think_valid or answer_valid,
            "has_both_tags": think_valid and answer_valid
        }
    
    def get_extraction_stats(self, extracted_answers: list) -> Dict:
        """Get statistics about answer extraction results.
        
        Args:
            extracted_answers: List of extracted answer dictionaries
        
        Returns:
            Dictionary with extraction statistics
        """
        total = len(extracted_answers)
        if total == 0:
            return {
                "total_responses": 0,
                "think_found_rate": 0.0,
                "answer_found_rate": 0.0,
                "both_found_rate": 0.0,
                "neither_found_rate": 0.0
            }
        
        think_found = sum(1 for answer in extracted_answers if answer["think_found"])
        answer_found = sum(1 for answer in extracted_answers if answer["answer_found"])
        both_found = sum(1 for answer in extracted_answers if answer["think_found"] and answer["answer_found"])
        neither_found = sum(1 for answer in extracted_answers if not answer["think_found"] and not answer["answer_found"])
        
        return {
            "total_responses": total,
                "think_found_rate": think_found / total,
            "answer_found_rate": answer_found / total,
            "both_found_rate": both_found / total,
            "neither_found_rate": neither_found / total
        }


def test_answer_extraction():
    """Test the answer extractor with various model outputs."""
    extractor = AnswerExtractor()
    
    # Test cases
    test_cases = [
        # Perfect output
        {
            "name": "Perfect output",
            "output": """<think>
I need to solve this step by step. First, I'll understand the problem.
</think>

<answer>
def solve_problem():
    return "solution"
</answer>"""
        },
        
        # Missing thinking tags
        {
            "name": "Missing thinking tags",
            "output": """<answer>
def solve_problem():
    return "solution"
</answer>"""
        },
        
        # Missing code tags
        {
            "name": "Missing code tags",
            "output": """<think>
I need to solve this step by step.
</think>"""
        },
        
        # Malformed tags (no closing)
        {
            "name": "Malformed tags",
            "output": """<think>
I need to solve this step by step.
<answer>
def solve_problem():
    return "solution"
"""
        },
        
        # No tags at all
        {
            "name": "No tags",
            "output": "Here is my solution: def solve(): return True"
        },
        
        # Tags in wrong order
        {
            "name": "Tags in wrong order",
            "output": """<answer>
def solve_problem():
    return "solution"
</answer>

<think>
I solved it by writing a function.
</thinking>"""
        },
        
        # Test pattern detection
        {
            "name": "Pattern detection test",
            "output": """<think>
I'll use sets and dicts to solve this efficiently.
</think>

<answer>
def solve_with_collections():
    # Using set() and dict() constructors
    unique_items = set()
    count_map = dict()
    
    for i in range(10):
        unique_items.add(i)
        count_map[i] = i * 2
    
    # Also test edge cases - these should NOT match
    offset = 5  # 'set' in offset
    predict = lambda x: x  # 'dict' in predict
    
    # But these should match
    my_set = set([1, 2, 3])
    my_dict = dict(a=1, b=2)
    
    return unique_items, count_map
</answer>"""
        }
    ]
    
    print("Testing Answer Extractor")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print("-" * 40)
        
        # Extract answer
        extracted = extractor.extract_answer(test_case['output'])
        
        # Validate tags
        validation = extractor.validate_tags(test_case['output'])
        
        print(f"Full output length: {len(extracted['full_output'])} chars")
        print(f"Thinking found: {extracted['think_found']}")
        print(f"Code found: {extracted['answer_found']}")
        print(f"Thinking content: {repr(extracted['think'][:50])}...")
        print(f"Code content: {repr(extracted['answer'][:50])}...")
        print(f"Tag validation: {validation}")
        if extracted['answer_found']:
            print(f"For loops: {extracted['for_loop_count']}")
            print(f"Function defs: {extracted['function_def_count']}")
            print(f"Curly braces: {extracted['curly_brace_count']}")
            print(f"set() calls: {extracted['set_constructor_count']}")
            print(f"dict() calls: {extracted['dict_constructor_count']}")
    
    # Test batch extraction
    outputs = [test_case['output'] for test_case in test_cases]
    batch_results = extractor.extract_batch_answers(outputs)
    stats = extractor.get_extraction_stats(batch_results)
    
    print(f"\nBatch extraction statistics:")
    print(f"Total responses: {stats['total_responses']}")
    print(f"Think found rate: {stats['think_found_rate']:.2%}")
    print(f"Answer found rate: {stats['answer_found_rate']:.2%}")
    print(f"Both found rate: {stats['both_found_rate']:.2%}")
    print(f"Neither found rate: {stats['neither_found_rate']:.2%}")


if __name__ == "__main__":
    test_answer_extraction() 