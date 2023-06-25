import pickle
import torch as T
import pandas as pd

from nltk.metrics.segmentation import pk, windowdiff
from src.utils import find_peak_left, find_peak_right, get_local_maxima, evaluation_config, apply_random_breaks
from src.dataset import PodcastDataset
from src.baseline import random_method, even_method, text_tiling_method


def flatten_features(batches_features):
    result = []
    for batch_features in batches_features:
        result += batch_features
    return result


def compute_window(features, start_index, end_index):
    """
    Given start and end index of embeddings, compute pooled window value.
    """
    stack = T.stack([feature for feature in features[start_index:end_index]])
    stack = stack.unsqueeze(0)
    stack_size = end_index - start_index
    pooling = T.nn.MaxPool2d((stack_size - 1, 1))
    return pooling(stack)


def create_blocks(features, k):
    """
    Compute comparison score for a gap (i).
    """
    block_features = []

    for i in range(k, len(features) - k):
        first_window_features = compute_window(features, i - k, i + 1)
        second_window_features = compute_window(features, i + 1, i + k + 2)
        block_features.append((first_window_features[0], second_window_features[0]))

    return block_features


def smooth(similarity_scores, smoothed_window_size, n):
    """
    This function applies a smoothing operation to a list of similarity scores.

    Parameters:
    similarity_scores (list): The list of similarity scores to be smoothed.
    smoothed_window_size (int): The size of the window to be used for the moving average operation.
    n (int): The number of times the smoothing operation is to be applied.

    Returns:
    list: The list of smoothed similarity scores.
    """
    smoothed_similarity_scores = similarity_scores[:]
    for _ in range(n):

        for i in range(len(smoothed_similarity_scores)):
            # Determine the start and end indices of the window for the moving average operation
            start = max(0, i - smoothed_window_size)
            end = min(len(similarity_scores) - 1, i + smoothed_window_size)

            # Extract the scores within the window
            neighbours = smoothed_similarity_scores[start:end]

            # Compute the average of the scores within the window
            smoothed_similarity_scores[i] = sum(neighbours) / len(neighbours)

    return smoothed_similarity_scores


def calc_similarity(windowed_block_features):
    """
    Given two sentence embeddings, compute the cosine similarity.
    """
    block_similarities = []

    for window_features in windowed_block_features:
        cosine = T.nn.CosineSimilarity()
        similarity = cosine(window_features[0], window_features[1])
        block_similarities.append(float(similarity))

    return block_similarities


def depth_score(smoothed_scores):
    """
    This function calculates the depth score for each index in the smoothed_scores.
    """
    depth_scores = []
    for i in range(1, len(smoothed_scores) - 1):
        left_peak = find_peak_left(smoothed_scores, i)
        right_peak = find_peak_right(smoothed_scores, i)
        depth_scores.append(
            (smoothed_scores[right_peak] - smoothed_scores[i]) + (smoothed_scores[left_peak] - smoothed_scores[i])
        )

    return depth_scores


def get_topic_change_indexes(depth_scores, threshold):
    """
    Convert depth score timeseries to topic change indexes.
    """
    if not depth_scores:
        return []

    local_maxima_indices, local_maxima = get_local_maxima(depth_scores)

    if not local_maxima:
        return []

    threshold = threshold * max(depth_scores)
    filtered_local_maxima_indices = [
        index for index, maxima in zip(local_maxima_indices, local_maxima)
        if maxima > threshold
    ]
    return filtered_local_maxima_indices


def compute_metrics(segmentation_pred_all, ground_truth_labels_all):
    pk_metric, windiff_metric = [], []

    for segmentation_pred, ground_truth_labels in zip(segmentation_pred_all, ground_truth_labels_all):

        predicted_labels = [0] * len(ground_truth_labels)
        for topic_change_position in segmentation_pred:
            predicted_labels[topic_change_position] = 1

        ground_truth_labels = "".join(map(str, ground_truth_labels))
        predicted_labels = "".join(map(str, predicted_labels))

        pk_metric.append(pk(ground_truth_labels, predicted_labels))

        k = int(round(len(ground_truth_labels) / (ground_truth_labels.count("1") * 2.0)))
        windiff_metric.append(windowdiff(ground_truth_labels, predicted_labels, k))

    avg_pk_metric = sum(pk_metric) / len(ground_truth_labels_all)
    avg_windiff_metric = sum(windiff_metric) / len(ground_truth_labels_all)

    return avg_pk_metric, avg_windiff_metric


def evaluate_topic_segmention(features_path, file_list):
    with open(features_path, 'rb') as f:
        features = pickle.load(f)

    podcast_dataset = PodcastDataset(file_list)
    labels = podcast_dataset.labels

    dfs = []
    current_pk = 1
    current_windiff = 1
    for index, config in enumerate(evaluation_config()):

        predictions = []
        for x in features:
            x = flatten_features(x)
            x = create_blocks(x, config.get("SENTENCE_COMPARISON_WINDOW"))
            x = calc_similarity(x)
            x = smooth(x, config.get("SMOOTHING_WINDOW"), config.get("SMOOTHING_ITERS"))
            x = depth_score(x)

            x_prediction = get_topic_change_indexes(x, config.get("TOPIC_CHANGE_THRESHOLD"))
            x_prediction.append(0)
            predictions.append(x_prediction)

        avg_pk, avg_windiff = compute_metrics(predictions, labels)

        config.update({
            "avg_pk": avg_pk,
            "avg_windiff": avg_windiff
        })

        if avg_pk < current_pk or avg_windiff < current_windiff:

            print(f"\n[{index}] Avg. Pk: {round(avg_pk, 3)}")
            print(f"[{index}] Avg. WinDiff: {round(avg_windiff, 3)}")

            current_pk = avg_pk
            current_windiff = avg_windiff

        dfs.append(pd.DataFrame.from_dict([config]))

    combined_df = pd.concat(dfs)

    ##################### BASELINES ####################################################################################

    def baseline_eval(method, n):
        avg_pk_all = 0
        avg_windiff_all = 0
        for _ in range(n):
            predicted_topic_change_positions_all = []
            for label in labels:
                predicted_topic_change_positions_all.append(method(label.tolist()))

            avg_pk, avg_windiff = compute_metrics(predicted_topic_change_positions_all, labels)

            avg_pk_all += avg_pk
            avg_windiff_all += avg_windiff

        print(f"Avg. Pk: {round(avg_pk_all / n, 3)}")
        print(f"Avg. WinDiff: {round(avg_windiff_all / n, 3)}")

    print("\n[Random]")
    baseline_eval(random_method(labels), 10)
    print("[Even]")
    baseline_eval(even_method(labels), 10)
    print("[TextTiling]")
    for episode in podcast_dataset.dataset:
        segmented_episode = apply_random_breaks(". ".join(episode[0]))
        baseline_eval(text_tiling_method(segmented_episode), 1)

    ##################### BASELINES ####################################################################################

    return combined_df


if __name__ == '__main__':
    output_path = 'data/output/eval_output_20230624.csv'

    output = evaluate_topic_segmention("data/features/features_20230624")
    output.to_csv(output_path)