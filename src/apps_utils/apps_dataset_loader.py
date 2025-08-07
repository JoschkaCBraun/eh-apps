"""
APPS Dataset Loader

This module provides functions to load and filter the cleaned APPS dataset
from Parquet files with various options for sampling and filtering.
"""

import pandas as pd
import numpy as np
import ast
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APPSDatasetLoader:
    """Load and filter the cleaned APPS dataset."""
    
    def __init__(self, data_dir: str = "data/APPS/cleaned"):
        self.data_dir = Path(data_dir)
        self.splits = ['train', 'eval', 'test']
        self.difficulties = ['introductory', 'interview', 'competition']
        
    def load_split(self, split: str) -> pd.DataFrame:
        """Load a specific split from Parquet file."""
        if split not in self.splits:
            raise ValueError(f"Split must be one of {self.splits}, got {split}")
            
        file_path = self.data_dir / f"apps_{split}.parquet"
        if not file_path.exists():
            raise FileNotFoundError(f"Split file not found: {file_path}")
            
        logger.info(f"Loading {split} split from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded {len(df)} problems from {split} split")
        
        return df
    
    def load_apps_samples(
        self,
        n_samples: int,
        split: str,
        difficulty: str = "all",
        min_test_cases: int = 0,
        max_test_cases: Optional[int] = None,
        has_solutions: Optional[bool] = None,
        has_starter_code: Optional[bool] = None,
        random_seed: Optional[int] = None
    ) -> List[Dict]:
        """
        Load and filter APPS dataset samples.
        
        Args:
            n_samples: Number of samples to load
            split: Dataset split ('train', 'eval', 'test', 'all')
            difficulty: Difficulty level ('introductory', 'interview', 'competition', 'all'). Default: 'all'
            min_test_cases: Minimum number of test cases required. Default: 0
            max_test_cases: Maximum number of test cases allowed. Default: None
            has_solutions: Filter by presence of solutions (True/False/None for no filter). Default: None
            has_starter_code: Filter by presence of starter code (True/False/None for no filter). Default: None
            random_seed: Random seed for reproducible sampling. Default: None
            
        Returns:
            List of problem dictionaries with recovered Python types for inputs/outputs
        """
        logger.info(f"Loading {n_samples} samples from {split} split")
        
        # Load data
        if split == "all":
            # Load all splits and combine
            all_data = []
            for s in self.splits:
                df = self.load_split(s)
                all_data.append(df)
            df = pd.concat(all_data, ignore_index=True)
        else:
            df = self.load_split(split)
        
        # Apply filters
        df = self._apply_filters(
            df, difficulty, min_test_cases, max_test_cases, 
            has_solutions, has_starter_code
        )
        
        if len(df) == 0:
            raise ValueError("No problems match the specified filters")
        
        # Sample data
        if n_samples > len(df):
            logger.warning(f"Requested {n_samples} samples but only {len(df)} available. Using all available samples.")
            n_samples = len(df)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        sampled_df = df.sample(n=n_samples, random_state=random_seed).reset_index(drop=True)
        
        # Convert to list of dictionaries
        problems = []
        for _, row in sampled_df.iterrows():
            problem = row.to_dict()
            
            # Always recover original data types for proper test execution
            problem = self._recover_data_types(problem)
            problems.append(problem)
        
        logger.info(f"Successfully loaded {len(problems)} problems")
        return problems
    
    def _apply_filters(
        self,
        df: pd.DataFrame,
        difficulty: str,
        min_test_cases: int,
        max_test_cases: Optional[int],
        has_solutions: Optional[bool],
        has_starter_code: Optional[bool]
    ) -> pd.DataFrame:
        """Apply filters to the dataset."""
        # Filter by difficulty
        if difficulty != "all":
            if difficulty not in self.difficulties:
                raise ValueError(f"Difficulty must be one of {self.difficulties}, got {difficulty}")
            df = df[df['difficulty'] == difficulty]
        
        # Filter by test case count
        df = df[df['n_test_cases'] >= min_test_cases]
        
        if max_test_cases is not None:
            df = df[df['n_test_cases'] <= max_test_cases]
        
        # Filter by solutions
        if has_solutions is not None:
            if has_solutions:
                df = df[df['n_solutions'] > 0]
            else:
                df = df[df['n_solutions'] == 0]
        
        # Filter by starter code
        if has_starter_code is not None:
            if has_starter_code:
                df = df[df['starter_code'].str.len() > 0]
            else:
                df = df[df['starter_code'].str.len() == 0]
        
        return df
    
    def _recover_single_type(self, data_str):
        """Recover a single data type from stringified format."""
        try:
            # Handle plain strings that aren't valid Python literals
            if isinstance(data_str, str):
                if data_str.startswith('"') and data_str.endswith('"'):
                    return data_str[1:-1]  # Remove quotes
                elif data_str.startswith("'") and data_str.endswith("'"):
                    return data_str[1:-1]  # Remove quotes
                else:
                    return ast.literal_eval(data_str)
            else:
                return data_str
        except (ValueError, SyntaxError):
            # If recovery fails, keep as string
            return data_str

    def _recover_data_types(self, problem: Dict) -> Dict:
        """Recover original data types from stringified inputs and outputs."""
        try:
            # Recover inputs
            if 'inputs' in problem and isinstance(problem['inputs'], list):
                problem['inputs'] = [self._recover_single_type(item) for item in problem['inputs']]
            
            # Recover outputs
            if 'outputs' in problem and isinstance(problem['outputs'], list):
                problem['outputs'] = [self._recover_single_type(item) for item in problem['outputs']]
                
        except Exception as e:
            logger.warning(f"Error recovering data types for problem {problem.get('problem_id', 'unknown')}: {e}")
        
        return problem
    
    def get_dataset_info(self) -> Dict:
        """Get information about the available dataset."""
        info = {}
        
        for split in self.splits:
            try:
                df = self.load_split(split)
                info[split] = {
                    'total_problems': len(df),
                    'difficulty_distribution': df['difficulty'].value_counts().to_dict(),
                    'test_case_stats': {
                        'mean': df['n_test_cases'].mean(),
                        'median': df['n_test_cases'].median(),
                        'min': df['n_test_cases'].min(),
                        'max': df['n_test_cases'].max()
                    },
                    'solution_stats': {
                        'mean': df['n_solutions'].mean(),
                        'median': df['n_solutions'].median(),
                        'min': df['n_solutions'].min(),
                        'max': df['n_solutions'].max()
                    },
                    'has_starter_code': (df['starter_code'].str.len() > 0).sum()
                }
            except FileNotFoundError:
                info[split] = {'error': 'File not found'}
        
        return info
    
    def print_dataset_info(self):
        """Print detailed information about the dataset."""
        info = self.get_dataset_info()
        
        print("\n" + "="*60)
        print("APPS DATASET INFORMATION")
        print("="*60)
        
        for split, split_info in info.items():
            if 'error' in split_info:
                print(f"\n{split.upper()}: {split_info['error']}")
                continue
                
            print(f"\n{split.upper()} SPLIT ({split_info['total_problems']:,} problems):")
            print("-" * 40)
            
            # Difficulty distribution
            for difficulty, count in split_info['difficulty_distribution'].items():
                percentage = (count / split_info['total_problems']) * 100
                print(f"  {difficulty:15}: {count:4,} problems ({percentage:5.1f}%)")
            
            # Test case statistics
            tc_stats = split_info['test_case_stats']
            print(f"\n  Test cases per problem:")
            print(f"    Mean: {tc_stats['mean']:.1f}")
            print(f"    Median: {tc_stats['median']:.1f}")
            print(f"    Range: {tc_stats['min']:.0f} - {tc_stats['max']:.0f}")
            
            # Solution statistics
            sol_stats = split_info['solution_stats']
            print(f"\n  Solutions per problem:")
            print(f"    Mean: {sol_stats['mean']:.1f}")
            print(f"    Median: {sol_stats['median']:.1f}")
            print(f"    Range: {sol_stats['min']:.0f} - {sol_stats['max']:.0f}")
            
            # Starter code
            starter_count = split_info['has_starter_code']
            starter_pct = (starter_count / split_info['total_problems']) * 100
            print(f"\n  Problems with starter code: {starter_count:,} ({starter_pct:.1f}%)")


def main():
    """Main function to test the dataset loader."""
    loader = APPSDatasetLoader()  # Uses default path
    
    # Print dataset information
    loader.print_dataset_info()
    
    # Test loading with different configurations
    print("\n" + "="*60)
    print("TESTING DATASET LOADER")
    print("="*60)
    
    # Test 1: Load 5 introductory problems from test split
    print("\nTest 1: Load 5 introductory problems from test split")
    problems = loader.load_apps_samples(
        n_samples=5,
        split="test",
        difficulty="introductory",
        random_seed=42
    )
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. Problem ID: {problem['problem_id']}")
        print(f"   Difficulty: {problem['difficulty']}")
        print(f"   Test cases: {len(problem['inputs'])}")
        print(f"   Solutions: {len(problem['solutions'])}")
        print(f"   Question preview: {problem['question'][:100]}...")
    
    # Test 2: Load problems with specific test case requirements
    print("\nTest 2: Load 3 problems with 3-5 test cases from eval split")
    problems = loader.load_apps_samples(
        n_samples=3,
        split="eval",
        min_test_cases=3,
        max_test_cases=5,
        random_seed=42
    )
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. Problem ID: {problem['problem_id']}")
        print(f"   Test cases: {len(problem['inputs'])}")
        print(f"   Inputs: {problem['inputs']}")
        print(f"   Outputs: {problem['outputs']}")


if __name__ == "__main__":
    main() 