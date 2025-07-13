#!/usr/bin/env python3
"""
Data Recipe Evaluation and Selection System

This script implements the evaluation framework for DataDecide methodology:
1. Analyze results from proxy training experiments
2. Rank data recipes by multiple performance metrics
3. Select optimal data mixtures for scaling to larger models
4. Generate recommendations for production training
"""

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Optional visualization imports
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False
    print("Visualization packages not available. Running evaluation only.")

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))


@dataclass
class RecipeEvaluation:
    """Evaluation results for a data recipe."""

    name: str
    description: str
    performance_score: float
    efficiency_score: float
    convergence_score: float
    overall_score: float
    ranking: int
    recommendations: List[str]


class DataRecipeEvaluator:
    """Evaluates and ranks data recipes based on proxy experiment results."""

    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()
        self.evaluations = []

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load proxy experiment results."""
        with open(self.results_file, "r") as f:
            return json.load(f)

    def evaluate_all_recipes(self) -> List[RecipeEvaluation]:
        """Evaluate all data recipes and return ranked list."""
        print("Evaluating Data Recipes")
        print("=" * 40)

        for result in self.results:
            evaluation = self._evaluate_single_recipe(result)
            self.evaluations.append(evaluation)

        # Sort by overall score
        self.evaluations.sort(key=lambda x: x.overall_score, reverse=True)

        # Assign rankings
        for i, eval_result in enumerate(self.evaluations):
            eval_result.ranking = i + 1

        return self.evaluations

    def _evaluate_single_recipe(self, result: Dict[str, Any]) -> RecipeEvaluation:
        """Evaluate a single data recipe."""
        # Performance Score (lower perplexity = better)
        performance_score = self._compute_performance_score(result)

        # Efficiency Score (faster convergence = better)
        efficiency_score = self._compute_efficiency_score(result)

        # Convergence Score (stable loss reduction = better)
        convergence_score = self._compute_convergence_score(result)

        # Overall Score (weighted combination)
        overall_score = (
            0.5 * performance_score  # Performance is most important
            + 0.3 * efficiency_score  # Efficiency matters for scaling
            + 0.2 * convergence_score  # Convergence indicates stability
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(result, performance_score, efficiency_score, convergence_score)

        return RecipeEvaluation(
            name=result["name"],
            description=result["description"],
            performance_score=performance_score,
            efficiency_score=efficiency_score,
            convergence_score=convergence_score,
            overall_score=overall_score,
            ranking=0,  # Will be set after sorting
            recommendations=recommendations,
        )

    def _compute_performance_score(self, result: Dict[str, Any]) -> float:
        """Compute performance score based on final perplexity."""
        eval_ppl = result["eval_perplexity"]
        train_ppl = result["train_perplexity"]

        # Normalize perplexity to 0-1 score (lower ppl = higher score)
        # Typical range for early training: 100-1000+ perplexity
        eval_score = max(0, min(1, (1000 - eval_ppl) / 900))
        train_score = max(0, min(1, (1000 - train_ppl) / 900))

        # Combine eval and train (eval is more important)
        return 0.7 * eval_score + 0.3 * train_score

    def _compute_efficiency_score(self, result: Dict[str, Any]) -> float:
        """Compute efficiency score based on convergence speed and training time."""
        conv_speed = result["convergence_speed"]
        training_time = result["training_time"]

        # Normalize convergence speed (higher = better)
        # Typical range: 0.001 - 0.1 loss reduction per step
        speed_score = max(0, min(1, conv_speed / 0.1))

        # Normalize training time (lower = better)
        # Typical range: 30-300 seconds for proxy experiments
        time_score = max(0, min(1, (300 - training_time) / 270))

        return 0.6 * speed_score + 0.4 * time_score

    def _compute_convergence_score(self, result: Dict[str, Any]) -> float:
        """Compute convergence score based on train/eval gap and stability."""
        train_loss = result["final_train_loss"]
        eval_loss = result["final_eval_loss"]

        # Smaller train/eval gap indicates better generalization
        gap = abs(eval_loss - train_loss)
        gap_score = max(0, min(1, (2.0 - gap) / 2.0))  # Penalty for gap > 2.0

        # Convergence speed indicates training stability
        conv_speed = result["convergence_speed"]
        stability_score = max(0, min(1, conv_speed / 0.05))  # Normalize to reasonable range

        return 0.6 * gap_score + 0.4 * stability_score

    def _generate_recommendations(
        self, result: Dict[str, Any], perf_score: float, eff_score: float, conv_score: float
    ) -> List[str]:
        """Generate actionable recommendations for a recipe."""
        recommendations = []

        recipe_name = result["name"]
        data_recipe = result["data_recipe"]

        # Performance-based recommendations
        if perf_score < 0.3:
            recommendations.append("Low performance - consider different data selection criteria")
        elif perf_score > 0.7:
            recommendations.append("High performance - good candidate for scaling")

        # Efficiency-based recommendations
        if eff_score < 0.3:
            recommendations.append("Slow convergence - may need longer training for larger models")
        elif eff_score > 0.7:
            recommendations.append("Fast convergence - efficient for rapid prototyping")

        # Recipe-specific recommendations
        if recipe_name == "random_baseline":
            recommendations.append("Baseline performance - compare other recipes to this")
        elif recipe_name == "high_quality":
            if perf_score > 0.5:
                recommendations.append("Quality filtering effective - consider for production")
            else:
                recommendations.append("Quality filtering not helping - check quality metrics")
        elif recipe_name == "high_diversity":
            if perf_score > 0.5:
                recommendations.append("Diversity selection working - good for generalization")
            else:
                recommendations.append("Diversity alone insufficient - combine with quality")
        elif recipe_name == "balanced_qd":
            recommendations.append("Balanced approach - good starting point for optimization")
        elif recipe_name == "financial_focus":
            if perf_score > 0.5:
                recommendations.append("Domain focus effective - consider for domain-specific models")
            else:
                recommendations.append("Domain focus limiting - use broader data mix")
        elif recipe_name == "long_context":
            if perf_score > 0.5:
                recommendations.append("Long context beneficial - important for complex reasoning")
            else:
                recommendations.append("Length threshold may be too restrictive")

        # Scaling recommendations
        if perf_score > 0.6 and eff_score > 0.5:
            recommendations.append("Excellent candidate for scaling to larger models")
        elif perf_score > 0.5:
            recommendations.append("Moderate candidate - test with medium subsample")
        else:
            recommendations.append("Poor candidate - requires recipe modification")

        return recommendations

    def print_evaluation_report(self):
        """Print comprehensive evaluation report."""
        if not self.evaluations:
            print("No evaluations available. Run evaluate_all_recipes() first.")
            return

        print("\nDataDecide Recipe Evaluation Report")
        print("=" * 60)

        # Summary statistics
        scores = [e.overall_score for e in self.evaluations]
        print(f"Total recipes evaluated: {len(self.evaluations)}")
        print(f"Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"Average score: {np.mean(scores):.3f}")

        # Detailed rankings
        print("\nDetailed Rankings:")
        print("-" * 60)

        for eval_result in self.evaluations:
            print(f"\n{eval_result.ranking}. {eval_result.name.upper()}")
            print(f"   Description: {eval_result.description}")
            print(f"   Overall Score: {eval_result.overall_score:.3f}")
            print(
                f"   Performance: {eval_result.performance_score:.3f}, "
                f"Efficiency: {eval_result.efficiency_score:.3f}, "
                f"Convergence: {eval_result.convergence_score:.3f}"
            )

            if eval_result.recommendations:
                print("   Recommendations:")
                for rec in eval_result.recommendations:
                    print(f"     • {rec}")

        # Top 3 recommendations
        print("\n" + "=" * 60)
        print("TOP 3 RECOMMENDATIONS FOR SCALING:")
        print("=" * 60)

        for i in range(min(3, len(self.evaluations))):
            eval_result = self.evaluations[i]
            print(f"\n{i + 1}. {eval_result.name} (Score: {eval_result.overall_score:.3f})")
            print(f"   {eval_result.description}")
            key_recs = [r for r in eval_result.recommendations if "scaling" in r.lower() or "production" in r.lower()]
            if key_recs:
                for rec in key_recs:
                    print(f"   → {rec}")

    def save_evaluation_results(self, output_file: str):
        """Save evaluation results to JSON file."""
        eval_data = []
        for eval_result in self.evaluations:
            eval_data.append(
                {
                    "name": eval_result.name,
                    "description": eval_result.description,
                    "ranking": eval_result.ranking,
                    "scores": {
                        "overall": eval_result.overall_score,
                        "performance": eval_result.performance_score,
                        "efficiency": eval_result.efficiency_score,
                        "convergence": eval_result.convergence_score,
                    },
                    "recommendations": eval_result.recommendations,
                }
            )

        with open(output_file, "w") as f:
            json.dump(eval_data, f, indent=2)

        print(f"\nEvaluation results saved to: {output_file}")

    def create_visualization(self, output_dir: str):
        """Create visualizations of evaluation results."""
        if not self.evaluations:
            print("No evaluations available for visualization.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Prepare data
        names = [e.name for e in self.evaluations]
        overall_scores = [e.overall_score for e in self.evaluations]
        perf_scores = [e.performance_score for e in self.evaluations]
        eff_scores = [e.efficiency_score for e in self.evaluations]
        conv_scores = [e.convergence_score for e in self.evaluations]

        # 1. Overall scores bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(names, overall_scores, color="skyblue", edgecolor="navy", alpha=0.7)
        plt.title("Data Recipe Overall Scores", fontsize=16, fontweight="bold")
        plt.ylabel("Overall Score", fontsize=12)
        plt.xlabel("Recipe Name", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", alpha=0.3)

        # Add score labels on bars
        for bar, score in zip(bars, overall_scores):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "recipe_scores.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Score breakdown radar chart
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection="polar"))

        # Take top 3 recipes for clarity
        top_3 = self.evaluations[:3]

        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = ["red", "blue", "green"]

        for i, eval_result in enumerate(top_3):
            values = [eval_result.performance_score, eval_result.efficiency_score, eval_result.convergence_score]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, color=colors[i], linewidth=2, label=eval_result.name)
            ax.fill(angles, values, color=colors[i], alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(["Performance", "Efficiency", "Convergence"])
        ax.set_ylim(0, 1)
        ax.set_title("Top 3 Recipes - Score Breakdown", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)

        plt.savefig(os.path.join(output_dir, "recipe_breakdown.png"), dpi=150, bbox_inches="tight")
        plt.close()

        print(f"Visualizations saved to: {output_dir}")


def main():
    """Main evaluation workflow."""
    print("DataDecide Recipe Evaluation System")
    print("=" * 50)

    # Check if results file exists
    results_file = "results/proxy_experiments/proxy_results.json"
    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        print("Run the proxy experiments first.")
        return

    # Initialize evaluator
    evaluator = DataRecipeEvaluator(results_file)

    # Run evaluation
    print(f"Loading results from: {results_file}")
    evaluations = evaluator.evaluate_all_recipes()

    # Print report
    evaluator.print_evaluation_report()

    # Save results
    output_dir = "results/recipe_evaluation"
    os.makedirs(output_dir, exist_ok=True)

    evaluator.save_evaluation_results(os.path.join(output_dir, "recipe_evaluations.json"))

    # Create visualizations
    evaluator.create_visualization(os.path.join(output_dir, "plots"))

    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
