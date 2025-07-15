"""Main training script for OLMo models."""

import argparse
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track

# No tokenizer imports - tokenization is handled by FinPileTokenizers
from data_decide.olmo.data.data_curation import DataDecideCurator
from data_decide.olmo.models.configuration_olmo import OLMO_CONFIGS
from data_decide.olmo.training.trainer import OLMoTrainer
from data_decide.olmo.utils.logging_utils import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train OLMo models")
    parser.add_argument(
        "--model_size",
        type=str,
        required=True,
        choices=list(OLMO_CONFIGS.keys()),
        help="Model size to train",
    )
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--config_file", type=str, required=True, help="Path to training config file")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--use_data_decide",
        action="store_true",
        default=True,
        help="Use DataDecide for data curation",
    )
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    args = parse_args()

    # Setup logging
    setup_logging(args.output_dir)
    logger.info(f"Starting OLMo training for {args.model_size} model")

    # Load configurations
    training_config = load_config(args.config_file)
    training_config["training"]["model_size"] = args.model_size
    training_config["output_dir"] = args.output_dir

    # Load data config
    data_config_path = Path(args.config_file).parent.parent / "data_configs" / "data_curation.yaml"
    data_config = load_config(str(data_config_path))

    # No tokenizer needed - FinPileTokenizers handles tokenization

    # Data curation with DataDecide
    if args.use_data_decide:
        console = Console()

        console.print("\n[bold cyan]Running DataDecide data curation...[/bold cyan]")
        curator = DataDecideCurator(
            data_path=args.data_path,
            tokenizer_name=None,  # Tokenization will be done by FinPileTokenizers
            proxy_model_size=data_config["data_curation"]["proxy_model_size"],
            num_proxy_steps=data_config["data_curation"]["proxy_training_steps"],
        )

        # Load and analyze data with progress
        with console.status("[bold green]Loading data..."):
            raw_data = curator.load_json_data()
            data_stats = curator.compute_data_statistics(raw_data)

        console.print(f"[green]✓[/green] Data loaded: {data_stats['total_documents']:,} documents")
        console.print(f"[green]✓[/green] Average length: {data_stats['avg_length']:.0f} characters")
        console.print(f"[green]✓[/green] Vocabulary size: {data_stats['vocabulary_size']:,}")

        # Create proxy experiments with progress
        console.print("\n[bold cyan]Creating proxy experiments...[/bold cyan]")
        proxy_datasets = []
        for i in track(range(data_config["data_curation"]["num_proxy_experiments"]), description="Creating datasets"):
            dataset = curator.create_proxy_experiments(raw_data, num_experiments=1)[0]
            proxy_datasets.append(dataset)

        # Evaluate and select best recipe with progress
        console.print("\n[bold cyan]Evaluating proxy datasets...[/bold cyan]")
        scores = {}
        for i, dataset in enumerate(track(proxy_datasets, description="Evaluating")):
            dataset_scores = curator.evaluate_proxy_datasets([dataset])
            scores.update(dataset_scores)

        best_recipe = curator.select_best_data_recipe(scores)
        console.print(f"\n[green]✓[/green] Selected best data recipe: {best_recipe}")

        # Display scores table
        from rich.table import Table

        table = Table(title="DataDecide Proxy Experiment Results")
        table.add_column("Dataset", style="cyan")
        table.add_column("Perplexity", style="magenta")
        table.add_column("Diversity", style="blue")
        table.add_column("Quality", style="green")
        table.add_column("Combined Score", style="bold yellow")

        for name, metrics in scores.items():
            table.add_row(
                name,
                f"{metrics['perplexity']:.2f}",
                f"{metrics['diversity']:.3f}",
                f"{metrics['quality']:.3f}",
                f"{metrics['combined_score']:.2f}",
            )

        console.print(table)

        # Use the best dataset
        train_dataset = proxy_datasets[int(best_recipe.split("_")[-1])]
    else:
        # Check if data is already tokenized with FinPileTokenizers

        if os.path.exists(f"{args.data_path}.bin") and os.path.exists(f"{args.data_path}.idx"):
            logger.info("Loading pre-tokenized FinPile dataset...")

            # Load using FinPileTokenizers format
            from data_decide.utils.finpile_data_loader import SimpleTapeDataset

            # Configure dataset parameters from config
            chunk_size = training_config.get("sequence_length", 4096)

            train_dataset = SimpleTapeDataset(
                prefix=args.data_path,
                chunk_size=chunk_size,
                eod_token_id=None,  # No packed sequences for now
            )

            logger.info(f"FinPile dataset loaded: {len(train_dataset):,} chunks of {chunk_size} tokens")

            # TODO: Add eval dataset support if needed
            eval_dataset = None

            # Initialize trainer with datasets
            trainer = OLMoTrainer(
                config=training_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=None,  # No tokenizer needed for pre-tokenized data
            )

            # Start training
            trainer.train()

            # Save final model
            trainer.save_model(os.path.join(args.output_dir, "final_model"))

            logger.info("Training completed successfully!")
            return
        else:
            # Raw data needs tokenization
            logger.error(
                "Raw data tokenization not yet implemented. Please pre-tokenize your data using FinPileTokenizers."
            )
            logger.error(
                "Run: python -m FinPileTokenizers.fsiltok.main --input <your_data> --prefix <output_path> --tokenizer <tokenizer_name>"
            )
            return

    # If we reach here, data_decide was used but we need pre-tokenized data
    logger.error("DataDecide requires pre-tokenized data. Please tokenize your curated data using FinPileTokenizers.")
    logger.error(
        "Run: python -m FinPileTokenizers.fsiltok.main --input <curated_data> --prefix <output_path> --tokenizer <tokenizer_name>"
    )


if __name__ == "__main__":
    main()
