"""
Chat Template Manager for GRPO Training

Implements the exact chat template structure from the Unsloth notebook,
with reasoning markers and system prompts for mathematical problem-solving.
"""

from typing import List, Dict, Optional


class ChatTemplateManager:
    """Manages chat templates for GRPO training following the notebook format."""
    
    def __init__(self):
        """Initialize template markers and system prompt."""
        # Exact markers from the notebook
        self.reasoning_start = "<start_working_out>"  # Acts as <think>
        self.reasoning_end = "<end_working_out>"      # Acts as </think>
        self.solution_start = "<SOLUTION>"
        self.solution_end = "</SOLUTION>"
        
        # System prompt from notebook
        self.system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {self.reasoning_start} and {self.reasoning_end}.
Then, provide your solution between {self.solution_start}{self.solution_end}"""
    
    def create_chat_template(self) -> str:
        """
        Create the Jinja2 chat template following the notebook structure.
        
        Returns:
            The chat template string with placeholders
        """
        template = (
            "{% if messages[0]['role'] == 'system' %}"
            "{{ messages[0]['content'] + eos_token }}"
            "{% set loop_messages = messages[1:] %}"
            "{% else %}"
            "{{ '{system_prompt}' + eos_token }}"
            "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
            "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '{reasoning_start}' }}"
            "{% endif %}"
        )
        
        # Replace placeholders with actual values
        template = template.replace("'{system_prompt}'", f"'{self.system_prompt}'")
        template = template.replace("'{reasoning_start}'", f"'{self.reasoning_start}'")
        
        return template
    
    def apply_to_tokenizer(self, tokenizer):
        """
        Apply the chat template to a tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer to apply template to
        """
        tokenizer.chat_template = self.create_chat_template()
        return tokenizer
    
    def format_messages(self, problem: str, solution: Optional[str] = None) -> List[Dict]:
        """
        Format a problem and optional solution into message format.
        
        Args:
            problem: The problem statement
            solution: Optional solution with reasoning and answer
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": problem}
        ]
        
        if solution is not None:
            messages.append({"role": "assistant", "content": solution})
        
        return messages
    
    def format_dataset_entry(self, problem: str, thoughts: str, answer: str) -> List[Dict]:
        """
        Format a dataset entry with problem, thoughts, and answer.
        
        Args:
            problem: The problem statement
            thoughts: The reasoning/working out (without tags)
            answer: The final answer
            
        Returns:
            List of message dictionaries with properly formatted solution
        """
        # Strip any existing tags from thoughts
        thoughts = thoughts.replace("<think>", "").replace("</think>", "")
        thoughts = thoughts.strip()
        
        # Construct the full solution with our tags
        solution = (
            f"{self.reasoning_start}{thoughts}{self.reasoning_end}"
            f"{self.solution_start}{answer}{self.solution_end}"
        )
        
        return self.format_messages(problem, solution)
    
    def format_for_generation(self, problem: str) -> List[Dict]:
        """
        Format a problem for generation (without solution).
        
        Args:
            problem: The problem statement
            
        Returns:
            List of message dictionaries ready for generation
        """
        return self.format_messages(problem, solution=None)
    
    def extract_solution_parts(self, generated_text: str) -> Dict[str, Optional[str]]:
        """
        Extract reasoning and solution from generated text.
        
        Args:
            generated_text: The model's generated response
            
        Returns:
            Dictionary with 'reasoning' and 'solution' keys
        """
        result = {"reasoning": None, "solution": None}
        
        # Try to extract reasoning
        if self.reasoning_start in generated_text and self.reasoning_end in generated_text:
            start_idx = generated_text.find(self.reasoning_start) + len(self.reasoning_start)
            end_idx = generated_text.find(self.reasoning_end)
            if start_idx < end_idx:
                result["reasoning"] = generated_text[start_idx:end_idx].strip()
        
        # Try to extract solution
        if self.solution_start in generated_text and self.solution_end in generated_text:
            start_idx = generated_text.find(self.solution_start) + len(self.solution_start)
            end_idx = generated_text.find(self.solution_end)
            if start_idx < end_idx:
                result["solution"] = generated_text[start_idx:end_idx].strip()
        
        return result
    
    def test_template(self, tokenizer):
        """
        Test the chat template with sample data.
        
        Args:
            tokenizer: Tokenizer with applied template
        """
        # Example from notebook
        test_messages = [
            {"role": "user", "content": "What is 1+1?"},
            {
                "role": "assistant", 
                "content": f"{self.reasoning_start}I think it's 2.{self.reasoning_end}"
                          f"{self.solution_start}2{self.solution_end}"
            },
            {"role": "user", "content": "What is 2+2?"},
        ]
        
        result = tokenizer.apply_chat_template(
            test_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        print("Template Test Result:")
        print("=" * 80)
        print(result)
        print("=" * 80)
        
        return result