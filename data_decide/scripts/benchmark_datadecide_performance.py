#!/usr/bin/env python3
"""
DataDecide Performance Benchmarking and Validation

This script provides comprehensive benchmarking and validation of the DataDecide methodology:
1. Analyze results across different dataset sizes and model sizes
2. Compare DataDecide against random baselines
3. Validate scaling predictions from proxy experiments
4. Generate performance reports and visualizations
"""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))


class DataDecideBenchmark:
    """Comprehensive benchmarking system for DataDecide methodology."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.proxy_results = None
        self.evaluation_results = None
        self.full_scale_results = []

    def load_all_results(self):
        """Load all available results from experiments."""
        print("Loading experimental results...")

        # Load proxy experiment results
        proxy_file = os.path.join(self.results_dir, "proxy_experiments/proxy_results.json")
        if os.path.exists(proxy_file):
            with open(proxy_file, "r") as f:
                self.proxy_results = json.load(f)
            print(f"Loaded {len(self.proxy_results)} proxy experiment results")

        # Load recipe evaluations
        eval_file = os.path.join(self.results_dir, "recipe_evaluation/recipe_evaluations.json")
        if os.path.exists(eval_file):
            with open(eval_file, "r") as f:
                self.evaluation_results = json.load(f)
            print(f"Loaded {len(self.evaluation_results)} recipe evaluations")

        # Load full-scale training results
        full_scale_pattern = os.path.join(self.results_dir, "full_scale_training*/comparison_results.json")
        for results_file in glob.glob(full_scale_pattern):
            with open(results_file, "r") as f:
                result = json.load(f)
                result["source_file"] = results_file
                self.full_scale_results.append(result)

        print(f"Loaded {len(self.full_scale_results)} full-scale experiment results")

    def analyze_proxy_experiment_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in proxy experiment results."""
        if not self.proxy_results:
            return {"error": "No proxy results available"}

        print("\nAnalyzing proxy experiment patterns...")

        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.proxy_results)

        analysis = {
            "recipe_performance": {},
            "convergence_patterns": {},
            "efficiency_metrics": {},
            "statistical_summary": {},
        }

        # Recipe performance analysis
        for recipe in df["name"].unique():
            recipe_data = df[df["name"] == recipe].iloc[0]
            analysis["recipe_performance"][recipe] = {
                "eval_perplexity": recipe_data["eval_perplexity"],
                "train_perplexity": recipe_data["train_perplexity"],
                "convergence_speed": recipe_data["convergence_speed"],
                "training_time": recipe_data["training_time"],
                "description": recipe_data["description"],
            }

        # Statistical summary
        analysis["statistical_summary"] = {
            "mean_eval_perplexity": df["eval_perplexity"].mean(),
            "std_eval_perplexity": df["eval_perplexity"].std(),
            "best_eval_perplexity": df["eval_perplexity"].min(),
            "worst_eval_perplexity": df["eval_perplexity"].max(),
            "mean_convergence_speed": df["convergence_speed"].mean(),
            "mean_training_time": df["training_time"].mean(),
        }

        # Identify best performing recipes
        best_recipe = df.loc[df["eval_perplexity"].idxmin()]
        analysis["best_recipe"] = {
            "name": best_recipe["name"],
            "eval_perplexity": best_recipe["eval_perplexity"],
            "improvement_over_mean": (df["eval_perplexity"].mean() - best_recipe["eval_perplexity"])
            / df["eval_perplexity"].mean()
            * 100,
        }

        print(
            f"Best recipe: {analysis['best_recipe']['name']} with perplexity {analysis['best_recipe']['eval_perplexity']:.2f}"
        )
        print(f"Improvement over average: {analysis['best_recipe']['improvement_over_mean']:.1f}%")

        return analysis

    def validate_scaling_predictions(self) -> Dict[str, Any]:
        """Validate how well proxy experiments predict full-scale performance."""
        if not self.proxy_results or not self.full_scale_results:
            return {"error": "Need both proxy and full-scale results for validation"}

        print("\nValidating scaling predictions...")

        validation = {"predictions": [], "correlation_metrics": {}, "accuracy_metrics": {}}

        # For each full-scale result, find corresponding proxy result
        for full_scale in self.full_scale_results:
            best_recipe_name = full_scale["best_recipe"]["name"]

            # Find proxy result for this recipe
            proxy_result = None
            for proxy in self.proxy_results:
                if proxy["name"] == best_recipe_name:
                    proxy_result = proxy
                    break

            if proxy_result:
                # Compare proxy vs full-scale performance
                proxy_ppl = proxy_result["eval_perplexity"]
                full_scale_ppl = full_scale["best_recipe"]["results"]["best_perplexity"]
                baseline_ppl = full_scale["baseline"]["results"]["best_perplexity"]

                # Calculate relative improvements
                proxy_improvement = (proxy_ppl - proxy_result["train_perplexity"]) / proxy_result["train_perplexity"]
                full_scale_improvement = (baseline_ppl - full_scale_ppl) / baseline_ppl

                validation["predictions"].append(
                    {
                        "recipe": best_recipe_name,
                        "proxy_perplexity": proxy_ppl,
                        "full_scale_perplexity": full_scale_ppl,
                        "baseline_perplexity": baseline_ppl,
                        "proxy_improvement": proxy_improvement,
                        "full_scale_improvement": full_scale_improvement,
                        "prediction_accuracy": abs(proxy_improvement - full_scale_improvement),
                    }
                )

        if validation["predictions"]:
            # Calculate correlation metrics
            proxy_scores = [p["proxy_improvement"] for p in validation["predictions"]]
            full_scale_scores = [p["full_scale_improvement"] for p in validation["predictions"]]

            if len(proxy_scores) > 1:
                correlation = np.corrcoef(proxy_scores, full_scale_scores)[0, 1]
                validation["correlation_metrics"]["improvement_correlation"] = correlation

            # Calculate prediction accuracy
            accuracies = [p["prediction_accuracy"] for p in validation["predictions"]]
            validation["accuracy_metrics"] = {
                "mean_prediction_error": np.mean(accuracies),
                "std_prediction_error": np.std(accuracies),
                "max_prediction_error": np.max(accuracies),
            }

        return validation

    def generate_performance_report(self, output_file: str):
        """Generate comprehensive performance report."""
        print(f"\nGenerating performance report: {output_file}")

        # Load all results
        self.load_all_results()

        # Run analyses
        proxy_analysis = self.analyze_proxy_experiment_patterns()
        scaling_validation = self.validate_scaling_predictions()

        # Generate report
        report = {
            "summary": {
                "total_proxy_experiments": len(self.proxy_results) if self.proxy_results else 0,
                "total_full_scale_experiments": len(self.full_scale_results),
                "best_recipe": proxy_analysis.get("best_recipe", {}),
                "methodology_validation": scaling_validation.get("correlation_metrics", {}),
            },
            "proxy_analysis": proxy_analysis,
            "scaling_validation": scaling_validation,
            "recommendations": self._generate_recommendations(proxy_analysis, scaling_validation),
        }

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        self._print_report_summary(report)

        return report

    def _generate_recommendations(self, proxy_analysis: Dict, scaling_validation: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []

        # Proxy experiment recommendations
        if proxy_analysis.get("best_recipe"):
            best_improvement = proxy_analysis["best_recipe"].get("improvement_over_mean", 0)
            if best_improvement > 10:
                recommendations.append(
                    f"Strong DataDecide signal: {best_improvement:.1f}% improvement over average. Scale to production."
                )
            elif best_improvement > 5:
                recommendations.append(
                    f"Moderate DataDecide signal: {best_improvement:.1f}% improvement. Test with larger datasets."
                )
            else:
                recommendations.append("Weak DataDecide signal. Refine data selection criteria.")

        # Scaling validation recommendations
        if scaling_validation.get("correlation_metrics"):
            correlation = scaling_validation["correlation_metrics"].get("improvement_correlation", 0)
            if correlation > 0.7:
                recommendations.append(
                    "High correlation between proxy and full-scale results. Proxy experiments are reliable."
                )
            elif correlation > 0.3:
                recommendations.append(
                    "Moderate correlation. Proxy experiments provide useful guidance but validate at scale."
                )
            else:
                recommendations.append(
                    "Low correlation. Proxy experiments may not predict full-scale performance well."
                )

        # Performance recommendations
        if len(self.full_scale_results) > 0:
            for result in self.full_scale_results:
                improvement = result.get("improvement_percent", 0)
                if improvement > 5:
                    recommendations.append(
                        f"DataDecide methodology shows {improvement:.1f}% improvement. Ready for production use."
                    )
                elif improvement > 0:
                    recommendations.append(
                        f"Small improvement ({improvement:.1f}%). Consider larger datasets or refined recipes."
                    )
                else:
                    recommendations.append("No improvement observed. Review data selection methodology.")

        # General recommendations
        recommendations.append("Continue testing with larger model sizes (150M, 450M) to validate scaling.")
        recommendations.append("Experiment with domain-specific data recipes for specialized models.")
        recommendations.append("Implement automated recipe optimization based on downstream task performance.")

        return recommendations

    def _print_report_summary(self, report: Dict):
        """Print a summary of the performance report."""
        print("\n" + "=" * 60)
        print("DATADECIDE PERFORMANCE REPORT SUMMARY")
        print("=" * 60)

        summary = report["summary"]

        print(f"Proxy Experiments: {summary['total_proxy_experiments']}")
        print(f"Full-Scale Experiments: {summary['total_full_scale_experiments']}")

        if summary.get("best_recipe"):
            best = summary["best_recipe"]
            print(f"\nBest Recipe: {best['name']}")
            print(f"Perplexity: {best['eval_perplexity']:.2f}")
            print(f"Improvement: {best['improvement_over_mean']:.1f}%")

        print("\nKey Recommendations:")
        for i, rec in enumerate(report["recommendations"][:3], 1):
            print(f"{i}. {rec}")

        print("\n" + "=" * 60)

    def create_visualizations(self, output_dir: str):
        """Create performance visualization plots."""
        if not self.proxy_results:
            print("No results available for visualization")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Proxy experiment results comparison
        df = pd.DataFrame(self.proxy_results)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Perplexity comparison
        recipes = df["name"].tolist()
        eval_ppls = df["eval_perplexity"].tolist()
        train_ppls = df["train_perplexity"].tolist()

        x = np.arange(len(recipes))
        width = 0.35

        ax1.bar(x - width / 2, eval_ppls, width, label="Eval Perplexity", alpha=0.8)
        ax1.bar(x + width / 2, train_ppls, width, label="Train Perplexity", alpha=0.8)
        ax1.set_xlabel("Recipe")
        ax1.set_ylabel("Perplexity")
        ax1.set_title("Perplexity by Recipe")
        ax1.set_xticks(x)
        ax1.set_xticklabels(recipes, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Convergence speed
        conv_speeds = df["convergence_speed"].tolist()
        colors = plt.cm.viridis(np.linspace(0, 1, len(recipes)))

        ax2.bar(recipes, conv_speeds, color=colors, alpha=0.8)
        ax2.set_xlabel("Recipe")
        ax2.set_ylabel("Convergence Speed")
        ax2.set_title("Convergence Speed by Recipe")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(axis="y", alpha=0.3)

        # Training time vs performance
        times = df["training_time"].tolist()
        ax3.scatter(times, eval_ppls, s=100, alpha=0.7, c=range(len(recipes)), cmap="viridis")
        ax3.set_xlabel("Training Time (s)")
        ax3.set_ylabel("Eval Perplexity")
        ax3.set_title("Training Time vs Performance")

        # Add recipe labels
        for i, recipe in enumerate(recipes):
            ax3.annotate(recipe[:8], (times[i], eval_ppls[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        ax3.grid(alpha=0.3)

        # Performance ranking
        df_sorted = df.sort_values("eval_perplexity")
        ranking_recipes = df_sorted["name"].tolist()
        ranking_ppls = df_sorted["eval_perplexity"].tolist()

        colors = [
            "gold" if i == 0 else "silver" if i == 1 else "chocolate" if i == 2 else "lightblue"
            for i in range(len(ranking_recipes))
        ]

        ax4.barh(ranking_recipes, ranking_ppls, color=colors, alpha=0.8)
        ax4.set_xlabel("Eval Perplexity")
        ax4.set_title("Recipe Performance Ranking")
        ax4.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "proxy_experiment_analysis.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # 2. Full-scale comparison (if available)
        if self.full_scale_results:
            self._create_full_scale_visualization(output_dir)

        print(f"Visualizations saved to: {output_dir}")

    def _create_full_scale_visualization(self, output_dir: str):
        """Create visualization for full-scale experiment results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        recipes = []
        best_ppls = []
        baseline_ppls = []
        improvements = []

        for result in self.full_scale_results:
            recipe_name = result["best_recipe"]["name"]
            best_ppl = result["best_recipe"]["results"]["best_perplexity"]
            baseline_ppl = result["baseline"]["results"]["best_perplexity"]
            improvement = result["improvement_percent"]

            recipes.append(recipe_name)
            best_ppls.append(best_ppl)
            baseline_ppls.append(baseline_ppl)
            improvements.append(improvement)

        # Perplexity comparison
        x = np.arange(len(recipes))
        width = 0.35

        ax1.bar(x - width / 2, baseline_ppls, width, label="Random Baseline", alpha=0.8, color="lightcoral")
        ax1.bar(x + width / 2, best_ppls, width, label="Best Recipe", alpha=0.8, color="skyblue")
        ax1.set_xlabel("Experiment")
        ax1.set_ylabel("Best Perplexity")
        ax1.set_title("Full-Scale Training: DataDecide vs Baseline")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"Exp {i + 1}" for i in range(len(recipes))])
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Improvement percentages
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        bars = ax2.bar(range(len(improvements)), improvements, color=colors, alpha=0.8)
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)
        ax2.set_xlabel("Experiment")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_title("DataDecide Improvement Over Baseline")
        ax2.set_xticks(range(len(improvements)))
        ax2.set_xticklabels([f"Exp {i + 1}" for i in range(len(improvements))])
        ax2.grid(axis="y", alpha=0.3)

        # Add improvement labels on bars
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{imp:+.1f}%",
                ha="center",
                va="bottom" if imp > 0 else "top",
            )

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "full_scale_comparison.png"), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    """Main benchmarking workflow."""
    print("DataDecide Performance Benchmarking System")
    print("=" * 50)

    results_dir = "results"
    output_dir = "results/benchmark"

    # Initialize benchmark system
    benchmark = DataDecideBenchmark(results_dir)

    # Generate comprehensive report
    report = benchmark.generate_performance_report(os.path.join(output_dir, "datadecide_performance_report.json"))

    # Create visualizations
    benchmark.create_visualizations(os.path.join(output_dir, "plots"))

    print(f"\nBenchmarking complete! Results available in: {output_dir}")


if __name__ == "__main__":
    main()
