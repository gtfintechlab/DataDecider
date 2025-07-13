#!/usr/bin/env python3
"""
Simple Data Recipe Evaluation System

This script provides basic evaluation of DataDecide proxy experiment results
without requiring visualization libraries.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))


class SimpleRecipeEvaluator:
    """Simple evaluator for data recipes based on proxy experiment results."""

    def __init__(self, results_file: str):
        self.results_file = results_file
        with open(results_file, "r") as f:
            self.results = json.load(f)

    def evaluate_and_rank_recipes(self) -> List[Dict[str, Any]]:
        """Evaluate all recipes and return ranked list."""
        print("Evaluating Data Recipes")
        print("=" * 40)

        evaluations = []

        for result in self.results:
            # Performance Score (lower perplexity = better)
            eval_ppl = result["eval_perplexity"]
            train_ppl = result["train_perplexity"]

            # Normalize perplexity to 0-1 score (handle very large values)
            eval_score = max(0, min(1, (1e15 - eval_ppl) / 1e15))
            train_score = max(0, min(1, (1e15 - train_ppl) / 1e15))
            performance_score = 0.7 * eval_score + 0.3 * train_score

            # Efficiency Score
            conv_speed = result["convergence_speed"]
            training_time = result["training_time"]

            speed_score = max(0, min(1, conv_speed / 0.6))  # Normalize convergence speed
            time_score = max(0, min(1, (60 - training_time) / 50))  # Prefer faster training
            efficiency_score = 0.6 * speed_score + 0.4 * time_score

            # Overall Score
            overall_score = 0.6 * performance_score + 0.4 * efficiency_score

            evaluation = {
                "name": result["name"],
                "description": result["description"],
                "eval_perplexity": eval_ppl,
                "train_perplexity": train_ppl,
                "convergence_speed": conv_speed,
                "training_time": training_time,
                "performance_score": performance_score,
                "efficiency_score": efficiency_score,
                "overall_score": overall_score,
                "raw_result": result,
            }

            evaluations.append(evaluation)

        # Sort by overall score (highest first)
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)

        # Add rankings
        for i, eval_result in enumerate(evaluations):
            eval_result["ranking"] = i + 1

        return evaluations

    def print_evaluation_report(self, evaluations: List[Dict[str, Any]]):
        """Print comprehensive evaluation report."""
        print("\nDataDecide Recipe Evaluation Report")
        print("=" * 60)

        # Summary statistics
        overall_scores = [e["overall_score"] for e in evaluations]
        eval_ppls = [e["eval_perplexity"] for e in evaluations]

        print(f"Total recipes evaluated: {len(evaluations)}")
        print(f"Overall score range: {min(overall_scores):.3f} - {max(overall_scores):.3f}")
        print(f"Average overall score: {np.mean(overall_scores):.3f}")
        print(f"Perplexity range: {min(eval_ppls):.2e} - {max(eval_ppls):.2e}")

        # Detailed rankings
        print("\nDetailed Rankings:")
        print("-" * 60)

        for eval_result in evaluations:
            print(f"\n{eval_result['ranking']}. {eval_result['name'].upper()}")
            print(f"   Description: {eval_result['description']}")
            print(f"   Overall Score: {eval_result['overall_score']:.3f}")
            print(f"   Eval Perplexity: {eval_result['eval_perplexity']:.2e}")
            print(
                f"   Performance: {eval_result['performance_score']:.3f}, "
                f"Efficiency: {eval_result['efficiency_score']:.3f}"
            )
            print(
                f"   Convergence Speed: {eval_result['convergence_speed']:.4f}, "
                f"Training Time: {eval_result['training_time']:.1f}s"
            )

        # Top 3 recommendations
        print("\n" + "=" * 60)
        print("TOP 3 RECOMMENDATIONS FOR SCALING:")
        print("=" * 60)

        for i in range(min(3, len(evaluations))):
            eval_result = evaluations[i]
            print(f"\n{i + 1}. {eval_result['name']} (Score: {eval_result['overall_score']:.3f})")
            print(f"   {eval_result['description']}")

            # Generate specific recommendations
            if eval_result["overall_score"] > 0.5:
                print("   → Excellent candidate for scaling to larger models")
            elif eval_result["overall_score"] > 0.3:
                print("   → Good candidate - test with medium subsample first")
            else:
                print("   → Needs improvement - consider refining selection criteria")

            if eval_result["convergence_speed"] > 0.55:
                print("   → Fast convergence - efficient for rapid experimentation")

            if eval_result["name"] == "high_quality":
                print("   → Quality-based selection effective - prioritize for production")
            elif eval_result["name"] == "financial_focus":
                print("   → Domain specialization working - good for financial models")
            elif eval_result["name"] == "balanced_qd":
                print("   → Balanced approach - good baseline for optimization")

        print("\n" + "=" * 60)

    def save_evaluation_results(self, evaluations: List[Dict[str, Any]], output_file: str):
        """Save evaluation results to JSON file."""
        # Remove raw_result to avoid duplication
        clean_evaluations = []
        for eval_result in evaluations.copy():
            clean_eval = {k: v for k, v in eval_result.items() if k != "raw_result"}
            clean_evaluations.append(clean_eval)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(clean_evaluations, f, indent=2)

        print(f"\nEvaluation results saved to: {output_file}")
        return output_file


def main():
    """Main evaluation workflow."""
    print("Simple DataDecide Recipe Evaluation System")
    print("=" * 50)

    # Check if results file exists
    results_file = "results/proxy_experiments/proxy_results.json"
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run the proxy experiments first.")
        return

    # Initialize evaluator
    evaluator = SimpleRecipeEvaluator(results_file)

    # Run evaluation
    print(f"Loading results from: {results_file}")
    evaluations = evaluator.evaluate_and_rank_recipes()

    # Print report
    evaluator.print_evaluation_report(evaluations)

    # Save results
    output_dir = "results/recipe_evaluation"
    output_file = evaluator.save_evaluation_results(evaluations, os.path.join(output_dir, "recipe_evaluations.json"))

    print("\nEvaluation complete! Next step: Run full-scale training with best recipe")
    print("Command: uv run python train_with_best_recipe.py")


if __name__ == "__main__":
    main()
