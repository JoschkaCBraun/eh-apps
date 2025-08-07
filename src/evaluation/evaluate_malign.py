#!/usr/bin/env python3
"""Simple evaluation script for malign outputs."""

import json
from pathlib import Path
from src.evaluation.answer_extractor import AnswerExtractor
from src.evaluation.code_executor import CodeExecutor
from src.utils.dataset_loader import APPSDatasetLoader
from datetime import datetime

# Load generation results
input_file = "data/generation_outputs/20250706_201930_eval_5problems_5models_outputs.json"
print(f"Loading generation results from {input_file}")
with open(input_file, 'r') as f:
    generation_results = json.load(f)

# Initialize components
answer_extractor = AnswerExtractor()
code_executor = CodeExecutor(timeout=5, max_memory_mb=100)
dataset_loader = APPSDatasetLoader(data_dir="data/APPS/cleaned")

# Load dataset to get test cases
print("Loading dataset...")
df = dataset_loader.load_split(generation_results["metadata"]["split"])
problems_dict = {}
for _, row in df.iterrows():
    problems_dict[row['problem_id']] = row.to_dict()

# Process each model
results = {}
for model_name, model_outputs in generation_results["results"].items():
    print(f"\nProcessing {model_name}...")
    model_results = []
    
    for output in model_outputs:
        result = {
            "problem_id": output["problem_id"],
            "api_success": output["api_success"],
            "api_error": output.get("api_error", "")
        }
        
        if output["api_success"]:
            # Extract code
            extracted = answer_extractor.extract_answer(output["model_output"])
            result["code_found"] = extracted["code_found"]
            result["thinking_found"] = extracted["thinking_found"]
            
            if extracted["code_found"]:
                # Get test cases
                problem = problems_dict.get(output["problem_id"])
                if problem and 'inputs' in problem and 'outputs' in problem:
                    test_cases = []
                    for inp, out in zip(problem['inputs'], problem['outputs']):
                        test_cases.append({'input': str(inp), 'output': str(out)})
                    
                    # Execute code
                    execution_result = code_executor.run_test_cases(extracted["code"], test_cases)
                    result["execution_success"] = execution_result["execution_success"]
                    result["pass_rate"] = execution_result["pass_rate"]
                    result["passed_count"] = execution_result["passed_count"]
                    result["total_count"] = execution_result["total_count"]
                else:
                    result["execution_success"] = False
                    result["pass_rate"] = 0.0
                    result["passed_count"] = 0
                    result["total_count"] = 0
            else:
                result["execution_success"] = False
                result["pass_rate"] = 0.0
                result["passed_count"] = 0
                result["total_count"] = 0
        else:
            result["code_found"] = False
            result["thinking_found"] = False
            result["execution_success"] = False
            result["pass_rate"] = 0.0
            result["passed_count"] = 0
            result["total_count"] = 0
            
        model_results.append(result)
    
    results[model_name] = model_results
    
    # Calculate summary stats
    api_success_rate = sum(r["api_success"] for r in model_results) / len(model_results)
    code_extraction_rate = sum(r["code_found"] for r in model_results) / len(model_results)
    execution_success_rate = sum(r["execution_success"] for r in model_results) / len(model_results)
    avg_pass_rate = sum(r["pass_rate"] for r in model_results) / len(model_results)
    
    print(f"  API Success: {api_success_rate:.2%}")
    print(f"  Code Found: {code_extraction_rate:.2%}")
    print(f"  Execution Success: {execution_success_rate:.2%}")
    print(f"  Avg Pass Rate: {avg_pass_rate:.2%}")

# Save results
output_data = {
    "metadata": {
        **generation_results["metadata"],
        "evaluation_timestamp": datetime.now().isoformat()
    },
    "results": results,
    "summary": {
        model_name: {
            "api_success_rate": sum(r["api_success"] for r in model_results) / len(model_results),
            "code_extraction_rate": sum(r["code_found"] for r in model_results) / len(model_results),
            "execution_success_rate": sum(r["execution_success"] for r in model_results) / len(model_results),
            "average_pass_rate": sum(r["pass_rate"] for r in model_results) / len(model_results)
        }
        for model_name, model_results in results.items()
    }
}

# Save to file
timestamp = generation_results["metadata"]["timestamp"].replace(":", "").replace("-", "")[:15]
output_file = f"data/scored_outputs/{timestamp}_eval_{generation_results['metadata']['n_problems']}problems_{len(results)}models_malign_scored.json"
print(f"\nSaving results to {output_file}")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2)

print("Evaluation complete!")