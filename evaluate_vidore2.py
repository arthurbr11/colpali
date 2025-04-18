import argparse
import json
import os
from typing import cast

import huggingface_hub
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from vidore_benchmark.evaluation.vidore_evaluators import ViDoReEvaluatorBEIR
from vidore_benchmark.retrievers import VisionRetriever

from colpali_engine.models import ColIdefics3, ColIdefics3Processor


def load_model_and_processor(model_dir: str, processor_dir: str = None, device: str = "cuda:0"):
    """
    Load the model and processor from the specified directory.

    Args:
        model_dir (str): Path to the model directory
        processor_dir (str): Path to the processor directory. If None, uses model_dir
        device (str): Device to load the model on

    Returns:
        tuple: (model, processor)
    """
    if processor_dir is None:
        processor_dir = model_dir

    processor = ColIdefics3Processor.from_pretrained(f"{processor_dir}-base")
    model = ColIdefics3.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    return model, processor


def evaluate_model(
    model_dir: str,
    output_dir: str = None,
    processor_dir: str = None,
    batch_sizes: dict = None,
):
    """
    Evaluate a model on the ViDoRe-2 benchmark.

    Args:
        model_dir (str): Path to the model directory
        output_dir (str): Directory to save results. If None, uses default structure
        processor_dir (str): Path to the processor directory. If None, uses model_dir
        batch_sizes (dict): Dictionary containing batch sizes for different operations
    """
    if batch_sizes is None:
        batch_sizes = {"query": 128, "passage": 128, "score": 128}

    # Get model name from directory
    model_name = model_dir.split("/")[-1]

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(model_dir, "metrics")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load model and processor
    model, processor = load_model_and_processor(model_dir, processor_dir)

    # Initialize retriever and evaluator
    vision_retriever = VisionRetriever(model=model, processor=processor)
    vidore_evaluator_beir = ViDoReEvaluatorBEIR(vision_retriever)

    # Get dataset collection
    collection = huggingface_hub.get_collection("vidore/vidore-benchmark-v2-67ae03e3924e85b36e7f53b0", token=False)
    dataset_names = [dataset_item.item_id for dataset_item in collection.items]

    # Evaluate on each dataset
    metrics_collection = {}
    for dataset_name in tqdm(dataset_names, desc="Evaluating dataset(s)"):
        print(f"\nEvaluating dataset: {dataset_name}")
        ds = {
            "corpus": cast(Dataset, load_dataset(dataset_name, name="corpus", split="test")),
            "queries": cast(Dataset, load_dataset(dataset_name, name="queries", split="test")),
            "qrels": cast(Dataset, load_dataset(dataset_name, name="qrels", split="test")),
        }
        metrics_collection[dataset_name] = vidore_evaluator_beir.evaluate_dataset(
            ds=ds,
            batch_query=batch_sizes["query"],
            batch_passage=batch_sizes["passage"],
            batch_score=batch_sizes["score"],
        )

    # Save detailed metrics
    output_file = os.path.join(output_dir, f"metrics_{model_name}.json")
    with open(output_file, "w") as f:
        json.dump(metrics_collection, f, indent=4)

    # Create summary DataFrame
    df = pd.DataFrame.from_dict(metrics_collection, orient="index")
    summary_file = os.path.join(output_dir, f"metrics_summary_{model_name}.csv")
    df.to_csv(summary_file)

    # Calculate and print average metrics
    print("\nAverage Metrics:")
    for metric in ["ndcg_at_5", "ndcg_at_10", "recall_at_5", "recall_at_10"]:
        if metric in df.columns:
            avg_value = df[metric].mean()
            print(f"Average {metric}: {avg_value:.4f}")

    return metrics_collection


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on ViDoRe-2 benchmark")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results (default: model_dir/metrics)"
    )
    parser.add_argument(
        "--processor_dir", type=str, default=None, help="Path to the processor directory (default: same as model_dir)"
    )
    parser.add_argument("--batch_query", type=int, default=128, help="Batch size for queries")
    parser.add_argument("--batch_passage", type=int, default=128, help="Batch size for passages")
    parser.add_argument("--batch_score", type=int, default=128, help="Batch size for scoring")

    args = parser.parse_args()

    batch_sizes = {"query": args.batch_query, "passage": args.batch_passage, "score": args.batch_score}

    evaluate_model(args.model_dir, args.output_dir, args.processor_dir, batch_sizes)


if __name__ == "__main__":
    main()
