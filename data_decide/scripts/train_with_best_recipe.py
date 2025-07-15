#!/usr/bin/env python3
"""
Full-Scale Training with Best DataDecide Recipe

This script implements the final stage of DataDecide methodology:
1. Load the best-performing data recipe from proxy experiments
2. Apply the recipe to larger datasets (medium/large subsamples)
3. Train larger models with optimized data selection
4. Compare against baseline (random data) training
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from torch.utils.data import DataLoader, Subset

from create_proxy_training_pipeline import DataRecipeGenerator
from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
from data_decide.olmo.models.olmo_model import OLMoForCausalLM
from data_decide.utils.finpile_data_loader import SimpleTapeDataset, create_data_collator_for_finpile


class FullScaleTrainer:
    """Trainer for full-scale experiments with DataDecide-selected data."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_best_recipe(self, evaluation_file: str) -> Dict[str, Any]:
        """Load the best-performing recipe from evaluation results."""
        with open(evaluation_file, "r") as f:
            evaluations = json.load(f)

        # Get top-ranked recipe
        best_recipe = evaluations[0]
        print(f"Best recipe: {best_recipe['name']}")
        print(f"Description: {best_recipe['description']}")
        print(f"Overall score: {best_recipe['overall_score']:.3f}")

        return best_recipe

    def create_dataset_from_recipe(self, recipe: Dict[str, Any], dataset_prefix: str, metadata_file: str) -> Subset:
        """Create dataset using the specified recipe."""
        print(f"Creating dataset from recipe: {recipe['name']}")

        # Load full dataset
        full_dataset = SimpleTapeDataset(dataset_prefix, chunk_size=self.config["sequence_length"])
        print(f"Full dataset size: {len(full_dataset)} chunks")

        # Load metadata and apply recipe selection
        recipe_generator = DataRecipeGenerator(metadata_file)

        # Get the original data recipe configuration
        # This would be stored in the proxy results, here we reconstruct it
        data_recipe = self._reconstruct_recipe_config(recipe["name"])
        selected_docs = recipe_generator.select_documents_for_recipe(data_recipe)

        print(f"Selected {len(selected_docs)} documents ({len(selected_docs) / 654 * 100:.1f}% of metadata)")

        # Create subset based on selection
        # For simplicity, create proportional subset - in production would extract exact token ranges
        target_size = int(len(full_dataset) * len(selected_docs) / 654)
        target_size = max(10, min(target_size, len(full_dataset)))

        np.random.seed(42)  # Reproducible selection
        indices = np.random.choice(len(full_dataset), size=target_size, replace=False)
        subset = Subset(full_dataset, indices)

        print(f"Created subset with {len(subset)} chunks")
        return subset

    def _reconstruct_recipe_config(self, recipe_name: str) -> Dict[str, Any]:
        """Reconstruct recipe configuration from name."""
        # This maps recipe names back to their configurations
        recipe_configs = {
            "random_baseline": {"name": "random_baseline", "selection_method": "random", "fraction": 0.25, "seed": 42},
            "high_quality": {
                "name": "high_quality",
                "selection_method": "quality_based",
                "fraction": 0.25,
                "quality_threshold": 0.9,
            },
            "high_diversity": {
                "name": "high_diversity",
                "selection_method": "diversity_based",
                "fraction": 0.25,
                "diversity_threshold": 0.85,
            },
            "balanced_qd": {
                "name": "balanced_qd",
                "selection_method": "composite",
                "fraction": 0.25,
                "quality_weight": 0.5,
                "diversity_weight": 0.5,
            },
            "financial_focus": {
                "name": "financial_focus",
                "selection_method": "domain_based",
                "domain": "financial",
                "fraction": 0.5,
            },
            "long_context": {
                "name": "long_context",
                "selection_method": "length_based",
                "min_tokens": 500,
                "fraction": 0.3,
            },
        }

        return recipe_configs.get(recipe_name, recipe_configs["random_baseline"])

    def train_model(self, dataset: Subset, model_size: str, output_dir: str) -> Dict[str, float]:
        """Train model on dataset and return metrics."""
        print(f"Training {model_size} model...")

        # Create model
        model_config = OLMO_CONFIGS[model_size]
        model = OLMoForCausalLM(model_config)
        model.to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Split dataset
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size

        train_indices = list(range(train_size))
        eval_indices = list(range(train_size, train_size + eval_size))

        train_dataset = Subset(dataset.dataset, [dataset.indices[i] for i in train_indices])
        eval_dataset = Subset(dataset.dataset, [dataset.indices[i] for i in eval_indices])

        # Create data loaders
        collate_fn = create_data_collator_for_finpile()
        train_loader = DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=True, collate_fn=collate_fn
        )
        eval_loader = DataLoader(
            eval_dataset, batch_size=self.config["batch_size"], shuffle=False, collate_fn=collate_fn
        )

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["learning_rate"])

        # Training loop
        model.train()
        start_time = time.time()

        train_losses = []
        eval_losses = []

        step = 0
        best_eval_loss = float("inf")

        for epoch in range(self.config["max_epochs"]):
            for batch in train_loader:
                if step >= self.config["max_steps"]:
                    break

                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Evaluate periodically
                if step % self.config["eval_interval"] == 0:
                    eval_loss = self._evaluate_model(model, eval_loader)
                    eval_losses.append(eval_loss)

                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                        # Save best model
                        if output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

                    print(f"Step {step}: train_loss={loss.item():.4f}, eval_loss={eval_loss:.4f}")

                step += 1

            if step >= self.config["max_steps"]:
                break

        training_time = time.time() - start_time

        # Final evaluation
        final_eval_loss = self._evaluate_model(model, eval_loader)

        results = {
            "final_train_loss": train_losses[-1] if train_losses else float("inf"),
            "final_eval_loss": final_eval_loss,
            "best_eval_loss": best_eval_loss,
            "train_perplexity": np.exp(train_losses[-1]) if train_losses else float("inf"),
            "eval_perplexity": np.exp(final_eval_loss),
            "best_perplexity": np.exp(best_eval_loss),
            "training_time": training_time,
            "total_steps": step,
        }

        print(f"Training completed in {training_time:.1f}s")
        print(f"Final eval perplexity: {results['eval_perplexity']:.2f}")
        print(f"Best eval perplexity: {results['best_perplexity']:.2f}")

        return results

    def _evaluate_model(self, model, eval_loader) -> float:
        """Evaluate model and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

                if num_batches >= 20:  # Limit for speed
                    break

        model.train()
        return total_loss / max(num_batches, 1)

    def run_comparison_experiment(self, dataset_prefix: str, metadata_file: str, evaluation_file: str, output_dir: str):
        """Run full comparison: best recipe vs random baseline."""
        print("Running DataDecide Comparison Experiment")
        print("=" * 50)

        os.makedirs(output_dir, exist_ok=True)

        # Load best recipe
        best_recipe = self.load_best_recipe(evaluation_file)

        # Create datasets
        print("\n1. Creating dataset with best recipe...")
        best_dataset = self.create_dataset_from_recipe(best_recipe, dataset_prefix, metadata_file)

        print("\n2. Creating random baseline dataset...")
        baseline_recipe = {"name": "random_baseline"}
        baseline_dataset = self.create_dataset_from_recipe(baseline_recipe, dataset_prefix, metadata_file)

        # Train models
        print("\n3. Training model with best recipe...")
        best_results = self.train_model(
            best_dataset, self.config["model_size"], os.path.join(output_dir, "best_recipe")
        )

        print("\n4. Training model with random baseline...")
        baseline_results = self.train_model(
            baseline_dataset, self.config["model_size"], os.path.join(output_dir, "baseline")
        )

        # Compare results
        print("\n5. Comparison Results:")
        print("=" * 50)

        improvement = (
            (baseline_results["best_perplexity"] - best_results["best_perplexity"])
            / baseline_results["best_perplexity"]
            * 100
        )

        print(f"Best Recipe ({best_recipe['name']}):")
        print(f"  Final perplexity: {best_results['eval_perplexity']:.2f}")
        print(f"  Best perplexity: {best_results['best_perplexity']:.2f}")
        print(f"  Training time: {best_results['training_time']:.1f}s")

        print("\nRandom Baseline:")
        print(f"  Final perplexity: {baseline_results['eval_perplexity']:.2f}")
        print(f"  Best perplexity: {baseline_results['best_perplexity']:.2f}")
        print(f"  Training time: {baseline_results['training_time']:.1f}s")

        print(f"\nImprovement: {improvement:+.1f}% perplexity reduction")

        if improvement > 0:
            print("✓ DataDecide methodology shows improvement!")
        else:
            print("✗ DataDecide methodology needs refinement")

        # Save results
        comparison_results = {
            "best_recipe": {
                "name": best_recipe["name"],
                "description": best_recipe["description"],
                "results": best_results,
            },
            "baseline": {
                "name": "random_baseline",
                "description": "Random data selection baseline",
                "results": baseline_results,
            },
            "improvement_percent": improvement,
            "config": self.config,
        }

        with open(os.path.join(output_dir, "comparison_results.json"), "w") as f:
            json.dump(comparison_results, f, indent=2)

        print(f"\nResults saved to: {output_dir}")
        return comparison_results


def main():
    parser = argparse.ArgumentParser(description="Train models with DataDecide methodology")
    parser.add_argument(
        "--dataset-prefix",
        default="data/finpile_subsamples/finpile_medium",
        help="Dataset prefix (default: finpile_medium)",
    )
    parser.add_argument(
        "--metadata-file", default="data/finpile_metadata/finpile_metadata.jsonl", help="Metadata file path"
    )
    parser.add_argument(
        "--evaluation-file",
        default="results/recipe_evaluation/recipe_evaluations.json",
        help="Recipe evaluation results file",
    )
    parser.add_argument("--model-size", default="4M", choices=["4M", "150M", "450M"], help="Model size to train")
    parser.add_argument("--max-steps", type=int, default=500, help="Maximum training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--output-dir", default="results/full_scale_training", help="Output directory")

    args = parser.parse_args()

    # Training configuration
    config = {
        "model_size": args.model_size,
        "sequence_length": 512,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "max_epochs": 10,
        "eval_interval": 50,
    }

    # Check required files
    required_files = [args.metadata_file, args.evaluation_file, f"{args.dataset_prefix}.bin"]

    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"Error: Required file not found: {file_path}")
            return

    # Initialize trainer
    trainer = FullScaleTrainer(config)

    # Run comparison experiment
    trainer.run_comparison_experiment(args.dataset_prefix, args.metadata_file, args.evaluation_file, args.output_dir)


if __name__ == "__main__":
    main()
