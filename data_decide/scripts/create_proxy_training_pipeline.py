#!/usr/bin/env python3
"""
Proxy Training Pipeline for DataDecide Methodology

This script implements the core DataDecide approach:
1. Create multiple data subset recipes
2. Train small proxy models on each recipe
3. Evaluate performance to predict which recipes will work best for larger models
4. Select optimal data mixtures based on proxy results
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from torch.utils.data import DataLoader, Subset

from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
from data_decide.olmo.models.olmo_model import OLMoForCausalLM
from data_decide.utils.finpile_data_loader import SimpleTapeDataset, create_data_collator_for_finpile


@dataclass
class ProxyExperimentConfig:
    """Configuration for a proxy training experiment."""

    name: str
    data_recipe: Dict[str, Any]  # Description of data subset/mixture
    model_size: str = "4M"
    max_steps: int = 100
    batch_size: int = 4
    learning_rate: float = 1e-4
    eval_steps: int = 20
    seed: int = 42


@dataclass
class ProxyResults:
    """Results from a proxy training experiment."""

    config: ProxyExperimentConfig
    final_train_loss: float
    final_eval_loss: float
    train_perplexity: float
    eval_perplexity: float
    convergence_speed: float  # Loss reduction per step
    training_time: float
    peak_memory_mb: float


class DataRecipeGenerator:
    """Generates different data recipes to test with DataDecide methodology."""

    def __init__(self, finpile_metadata_file: str):
        self.metadata_file = finpile_metadata_file
        self.documents = self._load_metadata()

    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load document metadata."""
        with open(self.metadata_file, "r") as f:
            return [json.loads(line) for line in f]

    def generate_recipes(self) -> List[Dict[str, Any]]:
        """Generate different data recipes to test."""
        recipes = []

        # Recipe 1: Random subset (baseline)
        recipes.append(
            {
                "name": "random_baseline",
                "description": "Random 25% of documents",
                "selection_method": "random",
                "fraction": 0.25,
                "seed": 42,
            }
        )

        # Recipe 2: High quality documents
        recipes.append(
            {
                "name": "high_quality",
                "description": "Top 25% by quality score",
                "selection_method": "quality_based",
                "fraction": 0.25,
                "quality_threshold": 0.9,
            }
        )

        # Recipe 3: High diversity documents
        recipes.append(
            {
                "name": "high_diversity",
                "description": "Top 25% by diversity score",
                "selection_method": "diversity_based",
                "fraction": 0.25,
                "diversity_threshold": 0.85,
            }
        )

        # Recipe 4: Balanced quality + diversity
        recipes.append(
            {
                "name": "balanced_qd",
                "description": "Balanced quality and diversity",
                "selection_method": "composite",
                "fraction": 0.25,
                "quality_weight": 0.5,
                "diversity_weight": 0.5,
            }
        )

        # Recipe 5: Financial domain focused
        recipes.append(
            {
                "name": "financial_focus",
                "description": "Financial domain documents only",
                "selection_method": "domain_based",
                "domain": "financial",
                "fraction": 0.5,  # Use more since it's domain-specific
            }
        )

        # Recipe 6: Long documents (better context)
        recipes.append(
            {
                "name": "long_context",
                "description": "Documents with >500 tokens",
                "selection_method": "length_based",
                "min_tokens": 500,
                "fraction": 0.3,
            }
        )

        return recipes

    def select_documents_for_recipe(self, recipe: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Select documents according to a recipe."""
        docs = self.documents.copy()

        if recipe["selection_method"] == "random":
            np.random.seed(recipe["seed"])
            selected_indices = np.random.choice(len(docs), size=int(len(docs) * recipe["fraction"]), replace=False)
            return [docs[i] for i in selected_indices]

        elif recipe["selection_method"] == "quality_based":
            docs.sort(key=lambda x: x["quality_score"], reverse=True)
            n_select = int(len(docs) * recipe["fraction"])
            return docs[:n_select]

        elif recipe["selection_method"] == "diversity_based":
            docs.sort(key=lambda x: x["diversity_score"], reverse=True)
            n_select = int(len(docs) * recipe["fraction"])
            return docs[:n_select]

        elif recipe["selection_method"] == "composite":
            # Compute composite score
            for doc in docs:
                doc["composite_score"] = (
                    recipe["quality_weight"] * doc["quality_score"]
                    + recipe["diversity_weight"] * doc["diversity_score"]
                )
            docs.sort(key=lambda x: x["composite_score"], reverse=True)
            n_select = int(len(docs) * recipe["fraction"])
            return docs[:n_select]

        elif recipe["selection_method"] == "domain_based":
            domain_docs = [d for d in docs if d["domain"] == recipe["domain"]]
            n_select = min(len(domain_docs), int(len(docs) * recipe["fraction"]))
            return domain_docs[:n_select]

        elif recipe["selection_method"] == "length_based":
            long_docs = [d for d in docs if d["token_count"] >= recipe["min_tokens"]]
            n_select = min(len(long_docs), int(len(docs) * recipe["fraction"]))
            return long_docs[:n_select]

        else:
            raise ValueError(f"Unknown selection method: {recipe['selection_method']}")


class ProxyTrainer:
    """Trains proxy models to evaluate data recipes."""

    def __init__(self, dataset_prefix: str):
        self.dataset_prefix = dataset_prefix
        self.full_dataset = SimpleTapeDataset(dataset_prefix, chunk_size=512)

    def create_subset_dataset(self, selected_docs: List[Dict[str, Any]]) -> Subset:
        """Create a subset dataset from selected documents."""
        # For simplicity, we'll create a random subset of the right size
        # In a full implementation, we'd extract exact token ranges
        target_size = int(len(self.full_dataset) * len(selected_docs) / 654)  # Scale by doc ratio
        target_size = max(10, min(target_size, len(self.full_dataset)))  # Ensure reasonable bounds

        indices = np.random.choice(len(self.full_dataset), size=target_size, replace=False)
        return Subset(self.full_dataset, indices)

    def train_proxy_model(self, config: ProxyExperimentConfig, selected_docs: List[Dict[str, Any]]) -> ProxyResults:
        """Train a proxy model on selected data and return results."""
        print(f"Training proxy model: {config.name}")
        print(f"  Selected documents: {len(selected_docs)}")
        print(f"  Model size: {config.model_size}")
        print(f"  Max steps: {config.max_steps}")

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create subset dataset
        subset_dataset = self.create_subset_dataset(selected_docs)
        print(f"  Dataset size: {len(subset_dataset)} chunks")

        # Split into train/eval
        train_size = int(0.8 * len(subset_dataset))
        eval_size = len(subset_dataset) - train_size
        train_indices = list(range(train_size))
        eval_indices = list(range(train_size, train_size + eval_size))

        train_dataset = Subset(subset_dataset.dataset, [subset_dataset.indices[i] for i in train_indices])
        eval_dataset = Subset(subset_dataset.dataset, [subset_dataset.indices[i] for i in eval_indices])

        # Create data loaders
        collate_fn = create_data_collator_for_finpile()
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

        # Create model
        model_config = OLMO_CONFIGS[config.model_size]
        model = OLMoForCausalLM(model_config)

        # Setup training
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

        # Training loop
        model.train()
        start_time = time.time()
        train_losses = []
        eval_losses = []

        step = 0
        for epoch in range(100):  # Large number, will break on max_steps
            for batch in train_loader:
                if step >= config.max_steps:
                    break

                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

                # Evaluate periodically
                if step % config.eval_steps == 0:
                    eval_loss = self._evaluate_model(model, eval_loader)
                    eval_losses.append(eval_loss)
                    print(f"    Step {step}: train_loss={loss.item():.4f}, eval_loss={eval_loss:.4f}")

                step += 1

            if step >= config.max_steps:
                break

        # Final evaluation
        final_train_loss = train_losses[-1] if train_losses else float("inf")
        final_eval_loss = self._evaluate_model(model, eval_loader)

        training_time = time.time() - start_time

        # Calculate convergence speed (loss reduction per step)
        if len(train_losses) > 10:
            initial_loss = np.mean(train_losses[:5])
            final_loss = np.mean(train_losses[-5:])
            convergence_speed = (initial_loss - final_loss) / len(train_losses)
        else:
            convergence_speed = 0.0

        results = ProxyResults(
            config=config,
            final_train_loss=final_train_loss,
            final_eval_loss=final_eval_loss,
            train_perplexity=np.exp(final_train_loss),
            eval_perplexity=np.exp(final_eval_loss),
            convergence_speed=convergence_speed,
            training_time=training_time,
            peak_memory_mb=torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0,
        )

        print(f"  Results: train_ppl={results.train_perplexity:.2f}, eval_ppl={results.eval_perplexity:.2f}")
        print(f"  Time: {training_time:.1f}s, Convergence: {convergence_speed:.6f}")

        return results

    def _evaluate_model(self, model, eval_loader) -> float:
        """Evaluate model and return average loss."""
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                outputs = model(**batch)
                total_loss += outputs.loss.item()
                num_batches += 1

                if num_batches >= 10:  # Limit eval for speed
                    break

        model.train()
        return total_loss / max(num_batches, 1)


def run_proxy_experiments(metadata_file: str, dataset_prefix: str, output_dir: str):
    """Run all proxy experiments and save results."""
    print("Running DataDecide Proxy Experiments")
    print("=" * 50)

    # Initialize components
    recipe_generator = DataRecipeGenerator(metadata_file)
    trainer = ProxyTrainer(dataset_prefix)

    # Generate recipes
    recipes = recipe_generator.generate_recipes()
    print(f"Generated {len(recipes)} data recipes")

    # Run experiments
    all_results = []

    for recipe in recipes:
        print(f"\nExperiment: {recipe['name']}")
        print(f"Description: {recipe['description']}")

        # Select documents for this recipe
        selected_docs = recipe_generator.select_documents_for_recipe(recipe)
        print(f"Selected {len(selected_docs)} documents")

        # Create experiment config
        config = ProxyExperimentConfig(
            name=recipe["name"],
            data_recipe=recipe,
            max_steps=50,  # Quick experiments
            batch_size=4,
            seed=42 + len(all_results),  # Different seed for each experiment
        )

        # Train proxy model
        try:
            results = trainer.train_proxy_model(config, selected_docs)
            all_results.append(results)
        except Exception as e:
            print(f"  Error in experiment {recipe['name']}: {e}")
            continue

    # Analyze and save results
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "proxy_results.json")

    # Convert results to serializable format
    results_data = []
    for result in all_results:
        results_data.append(
            {
                "name": result.config.name,
                "description": result.config.data_recipe["description"],
                "final_train_loss": result.final_train_loss,
                "final_eval_loss": result.final_eval_loss,
                "train_perplexity": result.train_perplexity,
                "eval_perplexity": result.eval_perplexity,
                "convergence_speed": result.convergence_speed,
                "training_time": result.training_time,
                "peak_memory_mb": result.peak_memory_mb,
                "data_recipe": result.config.data_recipe,
            }
        )

    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Print summary
    print("\nProxy Experiment Results Summary:")
    print("=" * 50)
    results_data.sort(key=lambda x: x["eval_perplexity"])

    for i, result in enumerate(results_data):
        print(
            f"{i + 1}. {result['name']}: eval_ppl={result['eval_perplexity']:.2f}, "
            f"conv_speed={result['convergence_speed']:.6f}"
        )

    # Identify best recipe
    best_result = results_data[0]
    print(f"\nBest recipe: {best_result['name']}")
    print(f"Description: {best_result['description']}")
    print(f"Eval perplexity: {best_result['eval_perplexity']:.2f}")

    return results_data


def main():
    print("DataDecide Proxy Training Pipeline")
    print("=" * 50)

    # Check inputs
    metadata_file = "data/finpile_metadata/finpile_metadata.jsonl"
    dataset_prefix = "data/finpile_subsamples/finpile_small"
    output_dir = "results/proxy_experiments"

    if not Path(metadata_file).exists():
        print(f"Error: Metadata file not found: {metadata_file}")
        print("Run test_datadecide_finpile.py first to create metadata.")
        return

    if not Path(f"{dataset_prefix}.bin").exists():
        print(f"Error: Dataset not found: {dataset_prefix}.bin")
        return

    # Run experiments
    results = run_proxy_experiments(metadata_file, dataset_prefix, output_dir)

    print(f"\nCompleted {len(results)} proxy experiments!")
    print("Next step: Scale best recipe to larger dataset and model.")


if __name__ == "__main__":
    main()
