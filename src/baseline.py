import random
from nltk.tokenize import TextTilingTokenizer


def random_method(ground_truth_labels):
    num_boundaries = ground_truth_labels.count(1)
    segmentation_pred = random.sample(range(len(ground_truth_labels)), num_boundaries)
    return sorted(segmentation_pred)


def even_method(ground_truth_labels):
    num_segments = ground_truth_labels.count(1) + 1
    segment_length = len(ground_truth_labels) // num_segments
    segmentation_pred = [i*segment_length for i in range(1, num_segments)]
    return segmentation_pred


def text_tiling_method(episode):
    ttt = TextTilingTokenizer()
    segments = ttt.tokenize(episode)
    boundaries = []
    length = 0
    for segment in segments[:-1]:
        length += len(segment.split('. '))
        boundaries.append(length)
    return boundaries