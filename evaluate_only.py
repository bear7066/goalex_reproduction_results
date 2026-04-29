import argparse
import json
import os
import numpy as np
from cluster_problem import ClusterProblem as Problem, ClusterProblemLabel as Label
import utils_performance

def evaluate(data_path, exp_dir):
    # Load data and labels
    with open(os.path.join(data_path, "data.json"), "r") as f:
        problem = Problem.from_json(f.read())
    
    label_path = os.path.join(data_path, "labels.json")
    if not os.path.exists(label_path):
        print(f"Error: labels.json not found in {data_path}")
        return

    with open(label_path, "r") as f:
        label = Label.from_json(f.read())

    # Load cluster results
    result_path = os.path.join(exp_dir, "cluster_result.json")
    if not os.path.exists(result_path):
        print(f"Error: cluster_result.json not found in {exp_dir}")
        return

    with open(result_path, "r") as f:
        cluster_result = json.load(f)

    # Reconstruct predicted labels
    # Map text to index for fast lookup
    text_to_idx = {t: i for i, t in enumerate(problem.texts)}
    
    n = len(problem.texts)
    predicted_labels = np.full(n, -1, dtype=int)
    
    descriptions = list(cluster_result.keys())
    
    for cluster_idx, description in enumerate(descriptions):
        texts_in_cluster = cluster_result[description]
        for text in texts_in_cluster:
            if text in text_to_idx:
                predicted_labels[text_to_idx[text]] = cluster_idx
            else:
                # Handle cases where text might be truncated or slightly different if processed
                pass

    # Filter out unmatched texts if necessary, or keep them as -1 (noise)
    # The utils_performance.get_cluster_performance expects matched arrays
    
    # We only evaluate on texts that exist in the label set (if subsampled)
    # Assuming data.json and labels.json are consistent
    
    ground_truth_labels = np.array(label.labels)
    
    # Handle subsampling if the result was run on a subset but we loaded full data
    # But usually exp_dir corresponds to the run. 
    # If the user ran with --subsample, the problem.texts loaded here (full) 
    # might be larger than what's in cluster_result.
    # However, predicted_labels is initialized to -1.
    
    # Let's assume standard usage where data.json matches the run scope 
    # OR we just evaluate on the intersection.
    
    # For robust evaluation, we should only consider indices that have a valid ground truth label
    # AND were part of the clustering problem.
    
    # Check if lengths match
    if len(ground_truth_labels) != len(predicted_labels):
        print(f"Warning: Length mismatch. GT: {len(ground_truth_labels)}, Pred: {len(predicted_labels)}")
        # If mismatch, we might need to rely on the user ensuring they point to the right data
        min_len = min(len(ground_truth_labels), len(predicted_labels))
        ground_truth_labels = ground_truth_labels[:min_len]
        predicted_labels = predicted_labels[:min_len]

    # Calculate metrics
    # Filter out -1 in predictions if we want to evaluate only clustered items?
    # Usually we treat -1 as a separate cluster or noise.
    # But get_cluster_performance uses sklearn metrics which handle noise differently depending on metric.
    # Let's follow experiment_recorder logic:
    # unmatched_text_indices = cluster_predictions == -1
    # labels[~unmatched_text_indices], cluster_predictions[~unmatched_text_indices]
    
    unmatched_mask = predicted_labels == -1
    matched_gt = ground_truth_labels[~unmatched_mask]
    matched_pred = predicted_labels[~unmatched_mask]
    
    print(f"Total samples: {len(ground_truth_labels)}")
    print(f"Unmatched samples: {np.sum(unmatched_mask)}")
    
    if len(matched_pred) == 0:
        print("No samples were clustered.")
        return

    nmi, ari, macro_f1, accuracy = utils_performance.get_cluster_performance(matched_gt, matched_pred)
    
    print("\nEvaluation Results (on matched texts):")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Adjusted Rand Index (ARI):    {ari:.4f}")
    print(f"Macro F1 Score:               {macro_f1:.4f}")
    print(f"Accuracy (Hungarian):         {accuracy:.4f}")

    # Print mapping
    true_descriptions = label.class_descriptions
    _, mapping = utils_performance.assign_labels(matched_gt, matched_pred)
    
    print("\nCluster Mapping:")
    for i in range(len(true_descriptions)):
        mapped_descs = [
            descriptions[p]
            for p in range(len(descriptions))
            if mapping.get(p, -1) == i
        ]
        print(f"GT: {true_descriptions[i]} <==> Pred: {mapped_descs}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GoalEx clustering results.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory containing data.json and labels.json")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory containing cluster_result.json")
    
    args = parser.parse_args()
    
    evaluate(args.data_path, args.exp_dir)
