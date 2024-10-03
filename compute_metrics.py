import json
from collections import defaultdict

import numpy as np
from metric import compute_bipartite_matching_metrics


def evaluate(ground_truth_path, prediction_path):
    target_count = 0
    target_type_counts = defaultdict(int)
    generated_count = 0
    generated_type_counts = defaultdict(int)
    target_entities_without_type = 0
    generated_entities_without_type = 0

    metrics = []
    counts = []
    nested_counts = []

    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    with open(prediction_path, "r") as f:
        predictions = json.load(f)

    new_dict = {}
    for k, v in enumerate(ground_truth.values()):
        new_dict[str(k)] = v
    ground_truth = new_dict

    new_dict = {}
    for k, v in enumerate(predictions.values()):
        new_dict[str(k)] = v
    predictions = new_dict

    print("Length of ground truth:", len(ground_truth))
    print("Length of predictions:", len(predictions))

    # Ensure both files have matching document IDs
    assert set(ground_truth.keys()) == set(
        predictions.keys()
    ), "Mismatch in document IDs between ground truth and predictions."

    exact_name_max = []
    exact_name_precision = []
    exact_name_recall = []
    approx_name_max = []
    approx_name_precision = []
    approx_name_recall = []
    multi_prop_max = []
    multi_prop_precision = []
    multi_prop_recall = []
    for doc_id, doc_data in ground_truth.items():
        target_entities = doc_data["entities"]
        target_count += len(target_entities)

        for entity in target_entities.values():
            try:
                target_type_counts[entity["type"]] += 1
            except KeyError:
                target_entities_without_type += 1

        generated_output = predictions[doc_id]["entities"]

        generated_entities = list(generated_output.values())
        generated_count += len(generated_entities)

        for entity in generated_entities:
            if "type" in entity:
                generated_type_counts[entity["type"]] += 1
            else:
                generated_entities_without_type += 1

        for normalization in ["Max", "Precision", "Recall"]:
            for measures in ["ExactName", "ApproxName", "MultiProp"]:
                (
                    final_metrics,
                    count,
                    nested_count,
                ) = compute_bipartite_matching_metrics(
                    target_entities,
                    generated_entities,
                    measures=measures,
                    normalization=normalization,
                    establish_threshold=0.6,
                )
                if measures == "ExactName":
                    if normalization == "Max":
                        exact_name_max.append(final_metrics["normalized_similarity"])
                    elif normalization == "Precision":
                        exact_name_precision.append(
                            final_metrics["normalized_similarity"]
                        )
                    elif normalization == "Recall":
                        exact_name_recall.append(final_metrics["normalized_similarity"])
                elif measures == "ApproxName":
                    if normalization == "Max":
                        approx_name_max.append(final_metrics["normalized_similarity"])
                    elif normalization == "Precision":
                        approx_name_precision.append(
                            final_metrics["normalized_similarity"]
                        )
                    elif normalization == "Recall":
                        approx_name_recall.append(
                            final_metrics["normalized_similarity"]
                        )
                elif measures == "MultiProp":
                    if normalization == "Max":
                        multi_prop_max.append(final_metrics["normalized_similarity"])
                        metrics.append(final_metrics)
                        counts.append(count)
                        nested_counts.append(nested_count)
                    elif normalization == "Precision":
                        multi_prop_precision.append(
                            final_metrics["normalized_similarity"]
                        )
                    elif normalization == "Recall":
                        multi_prop_recall.append(final_metrics["normalized_similarity"])
    # Compute and log metrics for generated text
    keys = set(key for d in metrics for key in d.keys())
    quantiles = [5, 10, 25, 50, 75, 90, 95]

    def compute_quantiles(data, quantiles):
        return {q: np.percentile(data, q) for q in quantiles}

    avg_metrics = {
        key: {
            "average": np.mean(
                [metric[key] for metric in metrics if key in metric.keys()]
            ),
            "quantiles": compute_quantiles(
                [metric[key] for metric in metrics if key in metric.keys()],
                quantiles,
            ),
            "raw_data": [metric[key] for metric in metrics if key in metric.keys()],
        }
        for key in keys
    }
    avg_metrics.update(
        {
            key
            + "_average": np.mean(
                [metric[key] for metric in metrics if key in metric.keys()]
            )
            for key in keys
        }
    )
    keys = set(key for d in counts for key in d.keys())
    total_metrics = {
        key: np.sum([count[key] for count in counts if key in count.keys()])
        for key in keys
    }

    outer_keys = set(key for d in nested_counts for key in d.keys())
    inner_keys = set(
        key
        for d in nested_counts
        for inner_dict in d.values()
        for key in inner_dict.keys()
    )

    total_nested_counts = {}
    for k in outer_keys:
        total_nested_counts[k] = {
            key: np.sum(
                [
                    count_dict[k][key]
                    for count_dict in nested_counts
                    if key in count_dict[k].keys()
                ]
            )
            for key in inner_keys
        }

    property_metrics = {}
    for k in inner_keys:
        property_metrics[k] = {
            "acc_token": (
                total_nested_counts["per_property_acc_token"][k]
                / total_nested_counts["key_matches"][k]
                if total_nested_counts["key_matches"][k] > 0
                else 0
            ),
            "acc_aon": (
                total_nested_counts["per_property_acc_aon"][k]
                / total_nested_counts["key_matches"][k]
                if total_nested_counts["key_matches"][k] > 0
                else 0
            ),
            "key_coverage": (
                total_nested_counts["key_matches"][k]
                / total_nested_counts["target_key_occurance"][k]
                if total_nested_counts["target_key_occurance"][k] > 0
                else 0
            ),
            "key_precision": (
                total_nested_counts["key_matches"][k]
                / total_nested_counts["pred_key_occurance"][k]
                if total_nested_counts["pred_key_occurance"][k] > 0
                else 0
            ),
        }
    print("Target Count:", target_count)
    print("Generated Count:", generated_count)
    print("Target Entities without type:", target_entities_without_type)
    print("Generated Entities without type:", generated_entities_without_type)

    avg_metrics.update(
        {
            # "avg_target_entities": avg_target_entities,
            # "target_type_counts": target_type_counts,
            # "avg_generated_entities": avg_generated_entities,
            # "generated_type_counts": generated_type_counts,
            # "avg_target_entities_without_type": avg_target_entities_without_type,
            # "avg_generated_entities_without_type": avg_generated_entities_without_type,
            "combined_coverage": total_metrics["established_entity_matches"]
            / total_metrics["target_entities_no_dup"],
            "combined_precision": total_metrics["established_entity_matches"]
            / total_metrics["predicted_output_entities_no_dup"],
        }
    )

    avg_metrics.update(property_metrics)

    result = {}
    result["exact_name_max"] = np.mean(exact_name_max)
    result["exact_name_precision"] = np.mean(exact_name_precision)
    result["exact_name_recall"] = np.mean(exact_name_recall)
    result["approx_name_max"] = np.mean(approx_name_max)
    result["approx_name_precision"] = np.mean(approx_name_precision)
    result["approx_name_recall"] = np.mean(approx_name_recall)
    result["multi_prop_max"] = np.mean(multi_prop_max)
    result["multi_prop_precision"] = np.mean(multi_prop_precision)
    result["multi_prop_recall"] = np.mean(multi_prop_recall)
    result["target_count"] = target_count
    result["generated_count"] = generated_count
    result["target_type_counts"] = target_type_counts
    result["generated_type_counts"] = generated_type_counts
    result["target_entities_without_type"] = target_entities_without_type
    result["generated_entities_without_type"] = generated_entities_without_type
    result.update(avg_metrics)
    with open(prediction_path[:-5] + "_metrics.json", "w") as f:
        json.dump(result, f, indent=4)
    return result
