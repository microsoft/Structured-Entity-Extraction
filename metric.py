import json
from copy import deepcopy

import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

# from utils import remove_duplicates_and_postprocess


def compute_distances(gt, pred, measures, weights_pks=None):
    """
    Args:
        gt: list of ground-truth entities, where each entity has
            - a 't' field consisting of the one-hot type torch.tensor
            - a 'pk' field consisting of multihot torch.tensor indicating the present pks
            - a 'pv' field containing a list of the property values, in an order consistent with the 'pk' field.
        pred: list of predicted entities
        measures: list of distance measures to apply
        weights_pks: torch.tensor defining a weight for each property key to compute weighted averages in the distance metrics.
    """
    assert measures in ["ExactName", "ApproxName", "MultiProp"]

    # if weights_pks is None:
    #     weights_pks = torch.ones_like(gt[0]["pk"])

    # def entity_type_distance_ce(e1, e2):
    #     t1, t2 = e1["t"].float(), e2["t"].float()
    #     epsilon = 1e-5
    #     t2 = torch.clamp(t2, min=epsilon)  # Ensure values are not too close to zero
    #     return -torch.sum(t1 * torch.log(t2))
    #
    # def entity_type_distance_acc(e1, e2):
    #     acc = (torch.argmax(e1["t"]) == torch.argmax(e2["t"])).float()
    #     return 1 - acc
    def entity_name_distance_approx(
        e1, e2, weights
    ):  # save as pv but only weights for name is 1 rest is 0
        v1, v2 = e1["pv"], e2["pv"]
        # Use jaccard similarity to compute the distance of the name
        # Split each property value into tokens (words)
        tokens_v1 = [
            set(value.lower().split())
            for index, value in enumerate(v1)
            if weights[index] == 1
        ]  # only for name
        tokens_v2 = [
            set(value.lower().split())
            for index, value in enumerate(v2)
            if weights[index] == 1
        ]  # only for name
        jaccard_similarities = []
        _weights = weights.clone()
        for i, (t1, t2) in enumerate(zip(tokens_v1, tokens_v2)):
            # Compute the Jaccard similarity for the token sets
            intersection_size = len(t1.intersection(t2))
            union_size = len(t1.union(t2))
            if union_size == 0:
                jaccard_sim = 0.0
                _weights[i] = 0.0
            else:
                jaccard_sim = intersection_size / union_size
            jaccard_similarities.append(jaccard_sim)
        dist = (
            1 - (torch.tensor(jaccard_similarities) * _weights).sum() / _weights.sum()
        )
        return dist

    def entity_name_distance_exact(
        e1, e2, weights
    ):  # save as pv but only weights for name is 1 rest is 0
        v1, v2 = e1["pv"], e2["pv"]
        v1 = [
            value for index, value in enumerate(v1) if weights[index] == 1
        ]  # only for name
        v2 = [
            value for index, value in enumerate(v2) if weights[index] == 1
        ]  # only for name
        matching = torch.tensor([int(a.lower() == b.lower()) for a, b in zip(v1, v2)])
        if len(matching) == 0:
            return 1.0
        else:
            return 1 - (matching.float() * weights).sum() / weights.sum()

    # def property_key_distance_bce(e1, e2):
    #     k1, k2 = e1["pk"].float(), e2["pk"].float()
    #     epsilon = 1e-5
    #     k2 = torch.clamp(
    #         k2, min=epsilon, max=1 - epsilon
    #     )  # Ensures values are between epsilon and 1-epsilon
    #     return -torch.sum(k1 * torch.log(k2) + (1 - k1) * torch.log(1 - k2))
    #
    # def property_key_distance_acc(e1, e2):
    #     k1, k2 = e1["pk"].float(), e2["pk"].float()
    #     k2_preds = (k2 >= 0.5).float()
    #     corrects = (k2_preds == k1).float().sum()
    #     acc = corrects / len(k1)
    #     return 1 - acc
    #
    # def property_value_prop_distance_acc(e1, e2, weights):
    #     v1, v2 = e1["pv"], e2["pv"]
    #
    #     matching = torch.tensor([int(a.lower() == b.lower()) for a, b in zip(v1, v2)])
    #
    #     return 1 - (matching.float() * weights).sum() / weights.sum()

    def property_value_token_distance_acc(e1, e2, weights):
        v1, v2 = e1["pv"], e2["pv"]

        # Split each property value into tokens (words)
        tokens_v1 = [set(value.lower().split()) for value in v1]
        tokens_v2 = [set(value.lower().split()) for value in v2]

        jaccard_similarities = []
        _weights = weights.clone()
        for i, (t1, t2) in enumerate(zip(tokens_v1, tokens_v2)):
            # Compute the Jaccard similarity for the token sets
            intersection_size = len(t1.intersection(t2))
            union_size = len(t1.union(t2))
            if union_size == 0:
                jaccard_sim = 0.0
                _weights[i] = 0.0
            else:
                jaccard_sim = intersection_size / union_size
            jaccard_similarities.append(jaccard_sim)
        dist = (
            1 - (torch.tensor(jaccard_similarities) * _weights).sum() / _weights.sum()
        )
        return dist

    distances = torch.zeros((len(gt), len(pred)))

    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            distance = 0
            if measures == "ExactName":
                distance = entity_name_distance_exact(g, p, weights_pks)
            elif measures == "ApproxName":
                distance = entity_name_distance_approx(g, p, weights_pks)
            elif measures == "MultiProp":  # We can also use bce here
                distance = property_value_token_distance_acc(g, p, weights_pks)
            # if "E-CE" in measures:
            #     distance += entity_type_distance_ce(g, p)
            # if "E-ACC" in measures:
            #     distance += entity_type_distance_acc(g, p)
            # if "Pk-BCE" in measures:
            #     distance += property_key_distance_bce(g, p)
            # if "Pk-ACC" in measures:
            #     distance += property_key_distance_acc(g, p)
            # if "Pv-prop-ACC" in measures:
            #     distance += property_value_prop_distance_acc(g, p, weights_pks)
            # if "Pv-token-ACC" in measures:
            #     distance += property_value_token_distance_acc(g, p, weights_pks)
            distances[i, j] = distance

    return distances


def bipartite_matching(distances):
    # Max is the maximum size of ground-truth set size and prediction set size
    # Precision is the size of the prediction set size
    # Recall is the size of the ground-truth set size
    biadjacency_matrix = csr_matrix(distances.numpy())
    # Add a constant (e.g., 1) to every distance to ensure no zero values
    biadjacency_matrix = biadjacency_matrix + csr_matrix(
        torch.ones_like(distances).numpy()
    )
    # print("biadjacency_matrix:", biadjacency_matrix.todense())
    row_ind, col_ind = min_weight_full_bipartite_matching(
        biadjacency_matrix, maximize=False
    )

    # Subtract the added constant for each matched pair
    min_num_entity = min(biadjacency_matrix.shape[0], biadjacency_matrix.shape[1])
    max_num_entity = max(biadjacency_matrix.shape[0], biadjacency_matrix.shape[1])
    matched_distances = biadjacency_matrix[row_ind, col_ind].sum() - min_num_entity
    # # optimal_metric_loss = (
    # #     (matched_distances + max_num_entity - min_num_entity) / max_num_entity
    # #     if max_num_entity != 0
    # #     else 0
    # # )
    # optimal_metric_loss = (matched_distances + denominator - min_num_entity) / denominator if denominator != 0 else 0
    # Obtain permutation
    permutation_ground_truth = torch.tensor(row_ind)[
        torch.argsort(torch.tensor(col_ind))
    ]
    permutation_prediction = torch.tensor(col_ind)
    return permutation_ground_truth, permutation_prediction
    # return optimal_metric_loss, permutation_ground_truth, permutation_prediction


def compute_bipartite_matching_metrics(
    target: list,
    predicted_output: list,
    measures,
    normalization,
    establish_threshold=0.6,
):
    """Compute metrics based on bipartite matching"""
    assert normalization in ["Max", "Precision", "Recall"]
    target = deepcopy(target)
    predicted_output = deepcopy(predicted_output)
    target_set_size = len(target)
    predicted_output_set_size = len(predicted_output)
    if isinstance(target, dict):
        target = [entity for entity in target.values()]
    # print("Prediction", predicted_output)
    # print("Target", target)
    keys = set(key for entity in target + predicted_output for key in entity.keys())
    # print("keys:", keys)
    keys = list(keys)

    # target = remove_duplicates_and_postprocess(target)
    # predicted_output = remove_duplicates_and_postprocess(predicted_output)

    # Pad the predicted entities or ground-truth with dummy entities
    #  to ensure that the number of entities is the same
    if target_set_size > predicted_output_set_size:
        predicted_output += [
            {} for _ in range(target_set_size - predicted_output_set_size)
        ]
    elif predicted_output_set_size > target_set_size:
        target += [{} for _ in range(predicted_output_set_size - target_set_size)]

    # print("Prediction", predicted_output)
    # print("Target", target)

    def get_key_index(key, keys):
        for i, k in enumerate(keys):
            if key == k:
                return i
        raise ValueError(f"The key {key} does not exists in the key list")

    # Create property key tensors
    def create_pk_tensor(entity, keys):
        tensor = [0] * len(keys)
        for key in entity.keys():
            tensor[get_key_index(key, keys)] = 1
        return torch.tensor(tensor)

    def create_pv_list(entity, keys):
        lst = [""] * len(keys)
        for key, value in entity.items():
            if not isinstance(value, str):
                if isinstance(value, list):
                    value = " ".join(value)
                else:
                    value = str(value)
            lst[get_key_index(key, keys)] = value

        return lst

    def jaccard_similarity(tokens_target, tokens_pred):
        # Compute the Jaccard similarity (intersection of the token set over the union)
        intersection_size = len(tokens_target.intersection(tokens_pred))
        union_size = len(tokens_target.union(tokens_pred))
        return intersection_size / union_size

    target_entities = [
        {"pk": create_pk_tensor(e, keys), "pv": create_pv_list(e, keys)} for e in target
    ]
    predicted_entities = [
        {"pk": create_pk_tensor(e, keys), "pv": create_pv_list(e, keys)}
        for e in predicted_output
    ]

    # Pad the predicted entities or ground-truth with dummy entities
    #  to ensure that the number of entities is the same
    # if target_set_size > predicted_output_set_size:
    #     predicted_entities += [
    #         {"pk": torch.zeros_like(target_entities[0]["pk"]), "pv": [""]}
    #         for _ in range(target_set_size - predicted_output_set_size)
    #     ]
    # elif predicted_output_set_size > target_set_size:
    #     target_entities += [
    #         {"pk": torch.zeros_like(predicted_entities[0]["pk"]), "pv": [""]}
    #         for _ in range(predicted_output_set_size - target_set_size)
    #     ]

    # assume a weight of 1 for each property apart from name
    try:
        weights = torch.zeros_like(target_entities[0]["pk"])
    except IndexError:
        print(target_entities)
        print(target)
        print(predicted_output)
    # try:
    #     if measures == "ExactName":
    #         weights[get_key_index("name", keys)] = 1
    #     elif measures == "ApproxName":
    #         weights[get_key_index("name", keys)] = 1
    #     elif measures == "MultiProp":
    #         weights[get_key_index("name", keys)] = 2
    #         for index, key in enumerate(keys):
    #             if key != "name":
    #                 weights[index] = 1
    # except ValueError:

    if measures == "ExactName":
        weights[get_key_index("entity name", keys)] = 1
    elif measures == "ApproxName":
        weights[get_key_index("entity name", keys)] = 1
    elif measures == "MultiProp":
        weights[get_key_index("entity name", keys)] = 11
        for index, key in enumerate(keys):
            if key != "entity name":
                weights[index] = 1
    # pv_distances_aon = compute_distances(
    #     target_entities, predicted_entities, ["Pv-prop-ACC"], weights
    # )
    # pv_distances_token = compute_distances(
    #     target_entities, predicted_entities, ["Pv-token-ACC"], weights
    # )
    # pv_distances_aon_unweighted = compute_distances(
    #     target_entities, predicted_entities, ["Pv-prop-ACC"], torch.ones_like(weights)
    # )
    # pv_distances_token_unweighted = compute_distances(
    #     target_entities, predicted_entities, ["Pv-token-ACC"], torch.ones_like(weights)
    # )
    # pk_distances_acc = compute_distances(
    #     target_entities, predicted_entities, ["Pk-ACC"], torch.ones_like(weights)
    # )

    # (
    #     pv_distances_token_loss,
    #     permutation_target,
    #     permutation_prediction,
    # ) = bipartite_matching(pv_distances_token)
    # pv_distances_aon_loss, _, _ = bipartite_matching(pv_distances_aon)
    # pk_distances_acc_loss, _, _ = bipartite_matching(pk_distances_acc)
    # pv_distances_token_unweighted_loss, _, _ = bipartite_matching(
    #     pv_distances_token_unweighted
    # )
    # pv_distances_aon_unweighted_loss, _, _ = bipartite_matching(
    #     pv_distances_aon_unweighted
    # )

    if measures == "ExactName":
        entity_distance = compute_distances(
            target_entities, predicted_entities, "ExactName", weights
        )
    elif measures == "ApproxName":
        entity_distance = compute_distances(
            target_entities, predicted_entities, "ApproxName", weights
        )
    elif measures == "MultiProp":
        entity_distance = compute_distances(
            target_entities, predicted_entities, "MultiProp", weights
        )

    # if normalization == "Max":
    #     (
    #         permutation_target,
    #         permutation_prediction,
    #     ) = bipartite_matching(entity_distance)
    # elif normalization == "Precision":
    #     (
    #         permutation_target,
    #         permutation_prediction,
    #     ) = bipartite_matching(entity_distance)
    # elif normalization == "Recall":
    (
        permutation_target,
        permutation_prediction,
    ) = bipartite_matching(entity_distance)

    # Only establish matches that have a distance below threshold
    #  the threshold and weight_pks is calibrated such that it does not suffice to have
    #  a matched type property without a matched name, if the entity only contains name and type (which is often the case)
    established_entity_matches = []
    established_entity_matches_tensor = []
    # print("permutation_target", permutation_target)
    # print("permutation_prediction", permutation_prediction)
    # print("target", target)
    # print("predicted_output", predicted_output)
    for predicted_idx, target_idx in enumerate(permutation_target):
        # if entity_distance[target_idx, predicted_idx] <= establish_threshold:
        #     established_entity_matches.append(
        #         (target[target_idx], predicted_output[predicted_idx])
        #     )
        # established_entity_matches.append(
        #     (target[target_idx], predicted_output[predicted_idx])
        # )
        established_entity_matches.append(
            (target[target_idx], predicted_output[predicted_idx])
        )
        established_entity_matches_tensor.append(
            (target_entities[target_idx], predicted_entities[predicted_idx])
        )

    def property_value_token_distance_acc(e1, e2, weights):
        v1, v2 = e1["pv"], e2["pv"]

        # Split each property value into tokens (words)
        tokens_v1 = [set(value.lower().split()) for value in v1]
        tokens_v2 = [set(value.lower().split()) for value in v2]

        jaccard_similarities = []
        _weights = weights.clone()
        for i, (t1, t2) in enumerate(zip(tokens_v1, tokens_v2)):
            # Compute the Jaccard similarity for the token sets
            intersection_size = len(t1.intersection(t2))
            union_size = len(t1.union(t2))
            if union_size == 0:
                jaccard_sim = 0.0
                _weights[i] = 0.0
            else:
                jaccard_sim = intersection_size / union_size
            jaccard_similarities.append(jaccard_sim)
        similarity = (
            torch.tensor(jaccard_similarities) * _weights
        ).sum() / _weights.sum()
        return similarity.item()

    # target_entities = [
    #     {"pk": create_pk_tensor(e, keys), "pv": create_pv_list(e, keys)} for e in target
    # ]
    # predicted_entities = [
    #     {"pk": create_pk_tensor(e, keys), "pv": create_pv_list(e, keys)}
    #     for e in predicted_output
    # ]

    # assume a weight of 1 for each property
    weights = torch.ones_like(target_entities[0]["pk"])

    all_similarities = []
    for t, p in established_entity_matches_tensor:
        similarity = property_value_token_distance_acc(t, p, weights)
        all_similarities.append(similarity)

    if normalization == "Max":
        normalized_similarity = sum(all_similarities) / max(
            target_set_size, predicted_output_set_size
        )
    elif normalization == "Precision":
        normalized_similarity = sum(all_similarities) / predicted_output_set_size
    elif normalization == "Recall":
        normalized_similarity = sum(all_similarities) / target_set_size

    per_property_acc_token = {key: 0.0 for key in keys}
    per_property_acc_aon = {key: 0.0 for key in keys}
    target_key_occurance = {key: 0.0 for key in keys}
    pred_key_occurance = {key: 0.0 for key in keys}
    key_matches = {key: 0.0 for key in keys}
    for e_target, e_pred in established_entity_matches:
        for key in keys:
            target_key_occurance[key] += key in e_target.keys()
            pred_key_occurance[key] += key in e_pred.keys()
            if key in e_target.keys() and key in e_pred.keys():
                key_matches[key] += 1.0
                tokens_target = set(e_target[key].lower().split())
                tokens_pred = set(e_pred[key].lower().split())

                jaccard_sim = jaccard_similarity(tokens_target, tokens_pred)
                per_property_acc_token[key] += jaccard_sim

                per_property_acc_aon[key] += (
                    e_target[key].lower() == e_pred[key].lower()
                )

        # # calculate per property similarity
        # prop_similarities = {}
        #
        # for pk in keys:
        #     prop_similarities[pk] = {}
        #     all_sim = []
        #     weights = torch.zeros_like(target_entities[0]["pk"])
        #     weights[get_key_index(pk, keys)] = 1
        #     for (t, p) in established_entity_matches_tensor:
        #         similarity = property_value_token_distance_acc(t, p, weights)
        #         all_sim.append(similarity)
        #     prop_similarities[pk]["Max"] = sum(all_sim) / max(target_set_size, predicted_output_set_size)
        #     prop_similarities[pk]["Precision"] = sum(all_sim) / predicted_output_set_size
        #     prop_similarities[pk]["Recall"] = sum(all_sim) / target_set_size
        #     if str(prop_similarities[pk]["Max"]) == "nan":
        #         prop_similarities[pk]["Max"] = 0
        #     if str(prop_similarities[pk]["Precision"]) == "nan":
        #         prop_similarities[pk]["Precision"] = 0
        #     if str(prop_similarities[pk]["Recall"]) == "nan":
        #         prop_similarities[pk]["Recall"] = 0

    counts_nested = {
        "per_property_acc_token": per_property_acc_token,
        "per_property_acc_aon": per_property_acc_aon,
        "target_key_occurance": target_key_occurance,
        "pred_key_occurance": pred_key_occurance,
        "key_matches": key_matches,
    }

    counts = {
        "established_entity_matches": len(established_entity_matches),
        "predicted_output_entities_no_dup": len(predicted_output),
        "target_entities_no_dup": len(target),
    }

    # bipartite_matching_metrics = {
    #     "normalized_similarity": normalized_similarity,
    #     # "pv_distances_token_loss": pv_distances_token_loss,
    #     # "pv_distances_aon_loss": pv_distances_aon_loss,
    #     # "pk_distances_acc_loss": pk_distances_acc_loss,
    #     # "pv_distances_token_unweighted_loss": pv_distances_token_unweighted_loss,
    #     # "pv_distances_aon_unweighted_loss": pv_distances_aon_unweighted_loss,
    # }
    final_metrics = {
        "normalized_similarity": normalized_similarity,
    }

    return final_metrics, counts, counts_nested


if __name__ == "__main__":
    # Sample usage

    dummy_entity = {
        "t": torch.tensor([0.0, 0.0]),
        "pk": torch.tensor([0.0, 0.0, 0.0]),
        "pv": ["dummy", "dummy", "dummy"],
    }

    # In this demo, we assume N=3 (max entity), M=2 (num entity type), K=3 (num property keys)
    # t (ground-truth) is a one-hot vector. pk (ground-truth) is a multi-hot vector.
    # t (prediction) is post-softmax. pk (prediction) is post-sigmoid.
    gt = [
        {
            "t": torch.tensor([1.0, 0.0]),
            "pk": torch.tensor([1.0, 1.0, 1.0]),
            "pv": ["XX apple", "round", "big"],
        },
        {
            "t": torch.tensor([0.0, 1.0]),
            "pk": torch.tensor([1.0, 0.0, 1.0]),
            "pv": ["YY banana", "long", "big"],
        },
        {
            "t": torch.tensor([1.0, 0.0]),
            "pk": torch.tensor([0.0, 1.0, 1.0]),
            "pv": ["ZZ grape", "round", "small"],
        },
    ]
    pred1 = [
        {
            "t": torch.tensor([1.0, 0.0]),
            "pk": torch.tensor([0.0, 1.0, 1.0]),
            "pv": ["ZZ grape", "round", "small"],
        },
        {
            "t": torch.tensor([1.0, 0.0]),
            "pk": torch.tensor([1.0, 1.0, 1.0]),
            "pv": ["XX apple", "round", "big"],
        },
        {
            "t": torch.tensor([0.0, 1.0]),
            "pk": torch.tensor([1.0, 0.0, 1.0]),
            "pv": ["YY banana", "long", "big"],
        },
    ]

    pred2 = [
        {
            "t": torch.tensor([0.999, 0.001]),
            "pk": torch.tensor([1.0, 1.0, 1.0]),
            "pv": ["YY apple", "round", "big"],
        },
        {
            "t": torch.tensor([0.001, 0.999]),
            "pk": torch.tensor([1.0, 0.1, 1.0]),
            "pv": ["XX banana", "long", "small"],
        },
        {
            "t": torch.tensor([0.8, 0.2]),
            "pk": torch.tensor([0.4, 0.9, 0.4]),
            "pv": ["ZZ grape", "round", "small"],
        },
    ]

    pred3 = [
        {
            "t": torch.tensor([0.7, 0.3]),
            "pk": torch.tensor([0.2, 0.8, 0.8]),
            "pv": ["XX peach", "round", "very small"],
        },
        dummy_entity,
        dummy_entity,
    ]

    pred_list = [pred1, pred2, pred3]
    for i in range(len(pred_list)):
        print(f"Compare GT with Pred{i + 1}:")
        # distances = compute_distances(gt, pred_list[i], measures=["E-CE"])
        # distances = compute_distances(gt, pred_list[i], measures=["Pk-BCE"])
        distances = compute_distances(gt, pred_list[i], measures=["E-ACC"])
        # distances = compute_distances(gt, pred_list[i], measures=["Pk-ACC"])
        # distances = compute_distances(gt, pred_list[i], measures=["Pv-prop-ACC"])
        # distances = compute_distances(gt, pred_list[i], measures=["Pv-token-ACC"])

        (
            optimal_metric_loss,
            permutation_ground_truth,
            permutation_prediction,
        ) = bipartite_matching(distances)
        print("optimal_metric_loss (CE loss or 1 - ACC):", optimal_metric_loss)
        print("permutation_ground_truth:", permutation_ground_truth)
        print("permutation_prediction:", permutation_prediction)
        print("-----")
