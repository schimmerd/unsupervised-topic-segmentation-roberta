import os
import itertools
import random

from pathlib import Path
from google.cloud import storage


def get_file_list(path):
    path_to_data = Path(path)
    return [str(x) for x in path_to_data.glob("*.hdf5") if x.is_file()]


def upload_features(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client(os.environ.get("project_id"))
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def find_peak_left(smoothed_scores, index):
    """
    This function finds the peak on the left side of the given index.
    """
    left = index - 1
    while left > 0 and smoothed_scores[left - 1] > smoothed_scores[left]:
        left -= 1
    return left


def find_peak_right(smoothed_scores, index):
    """
    This function finds the peak on the right side of the given index.
    """
    right = index + 1
    while right < (len(smoothed_scores) - 1) and smoothed_scores[right + 1] > smoothed_scores[right]:
        right += 1
    return right


def get_local_maxima(array):
    """
    Get the local maxima from an array.
    """
    local_maxima_indices = []
    local_maxima_values = []
    for i in range(1, len(array) - 1):
        if array[i - 1] < array[i] and array[i] > array[i + 1]:
            local_maxima_indices.append(i)
            local_maxima_values.append(array[i])
    return local_maxima_indices, local_maxima_values


def evaluation_config():
    SENTENCE_COMPARISON_WINDOW: list[int] = [10, 15, 20]
    SMOOTHING_ITERS: list[int] = [2, 4, 6]
    SMOOTHING_WINDOW: list[int] = [1, 3, 5]
    TOPIC_CHANGE_THRESHOLD: list[float] = [0.6, 0.7, 0.8]

    eval_config = list(itertools.product(
        SENTENCE_COMPARISON_WINDOW,
        SMOOTHING_ITERS,
        SMOOTHING_WINDOW,
        TOPIC_CHANGE_THRESHOLD)
    )

    for ec in eval_config:
        yield {
            "SENTENCE_COMPARISON_WINDOW": ec[0],
            "SMOOTHING_ITERS": ec[1],
            "SMOOTHING_WINDOW": ec[2],
            "TOPIC_CHANGE_THRESHOLD": ec[3]
        }


def apply_random_breaks(raw_text):
    text_length = len(raw_text)
    segment_indices = sorted(random.sample(range(text_length), random.randint(1, 10)))
    segmented_text = ''
    last_index = 0
    for index in segment_indices:
        segmented_text += raw_text[last_index:index] + '\n\n'
        last_index = index
    segmented_text += raw_text[last_index:]
    return segmented_text
