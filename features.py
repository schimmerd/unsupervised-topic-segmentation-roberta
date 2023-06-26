import pickle
import torch as T

from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer

from src.dataset import PodcastDataset
from src.utils import get_file_list, upload_features
from config import device, data_path, source_features_path, destination_feature_path, bucket_name


model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model.to(device)


def compute_roberta_embeddings(sentences, layer=-2):
    """
    Compute RoBERTa embeddings for each sentence in the given DataFrame.

    Parameters:
        sentences (pd.DataFrame): The DataFrame containing the sentences.
        layer (int): Default -2
    Returns:
        array: The RoBERTa embeddings for each sentence.
    """
    embeddings = []

    for sentence in sentences:
        tensor_tokens = tokenizer.encode(sentence, return_tensors='pt', truncation=True, padding=True)
        tensor_tokens = tensor_tokens.to(device)

        with T.no_grad():
            output = model(input_ids=tensor_tokens)

        hidden_states = output[2]
        pooling = T.nn.AvgPool2d((len(tensor_tokens[0]), 1))
        sentence_features = pooling(hidden_states[layer])
        embeddings.append(sentence_features[0])

    return embeddings


def save_features():
    model.eval()

    podcast = PodcastDataset(get_file_list(data_path))

    features = []
    for episode in tqdm(podcast.dataset, desc="Compute sentence embeddings"):
        features.append(compute_roberta_embeddings(episode[0]))

    with open(source_features_path, 'wb') as f:
        pickle.dump(features, f)
    # upload_features(bucket_name, source_features_path, destination_feature_path)
