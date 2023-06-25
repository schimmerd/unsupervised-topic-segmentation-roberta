import pickle
import torch as T

from transformers import RobertaModel, RobertaTokenizer
from pathlib import Path
from src.dataset import PodcastDataset


model = RobertaModel.from_pretrained('roberta-base', output_hidden_states=True)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

device = T.device("cuda" if T.cuda.is_available() else "cpu")
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


def save_features(input_file_path, output_file_path):
    model.eval()

    podcast = PodcastDataset(input_file_path)

    features = []
    for episode in podcast.dataset:
        features.append(compute_roberta_embeddings(episode[0]))

    with open(output_file_path, 'wb') as file:
        pickle.dump(features, file)


if __name__ == '__main__':
    path_to_data = Path("data/podcast")
    files = [str(x) for x in path_to_data.glob("*.hdf5") if x.is_file()]
    save_features(files, 'data/features/features_20230624')
