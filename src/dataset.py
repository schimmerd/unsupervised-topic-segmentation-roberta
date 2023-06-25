import h5py
import string
import numpy as np


filler_words = [
    'uhm', 'um', 'uh', 'oh', 'ou', 'yea', 'yeah', 'like', 'you know', 'so',
    'actually', 'basically', 'seriously', 'literally', 'anyway', 'sort of',
    'kind of', 'well', 'I mean', 'right', 'you see'
]


class PodcastDataset:
    def __init__(self, file_list, stop_words=None):
        if stop_words is None:
            stop_words = filler_words

        self.file_list = file_list
        self.stop_words = stop_words

        self.dataset = []
        self.labels = []
        for idx, (text, labels) in enumerate(self._extract_data()):
            episode = self._decode_episode(text)
            episode = self._cleansing(episode)
            self.dataset.append([episode, labels])
            self.labels.append([labels])

    def _extract_data(self):
        for file in self.file_list:
            with h5py.File(file, 'r') as hf:
                groups = [item[1] for item in hf.items()]

                docs = [grp['sents'][()] for grp in groups]
                docs = [docs.tolist() for docs in docs]
                labels = np.array([grp['labels'][()] for grp in groups], dtype=object)

                docs = [x for x in docs if len(x) > 0]
                labels = [x for x in labels if len(x) > 0]

                for doc, label in zip(docs, labels):
                    assert len(doc) == len(label)
                    yield doc, label

    @staticmethod
    def _decode_episode(episode):
        """
        Decodes text input

        Parameters:
            episode (list): A list of binary encoded sentences in the episode.

        Returns:
            decoded_episode (list): A list of human-readable texts
        """
        decoded_episode = []
        for sentence in episode:
            decoded_episode.append(sentence.decode("UTF-8"))
        return decoded_episode

    def _cleansing(self, episode):
        """
        Cleanses an episode by removing punctuation, converting to lower case,
        and removing stop words from each sentence in the episode.

        Parameters:
            episode (list): A list of sentences in the episode.

        Returns:
            cleaned_episode (list): A list of cleansed sentences.
        """
        cleaned_episode = []

        for sentence in episode:

            sentence_without_punctuation = sentence.translate(str.maketrans('', '', string.punctuation))

            # Tokenize the sentence, convert to lower case, and remove stop words
            tokens = [
                token.lower()
                for token in sentence_without_punctuation.split()
                if token.lower() not in self.stop_words
            ]

            # Only add sentences that have more than 2 words after cleansing
            if len(tokens) > 2:
                cleaned_sentence = " ".join(tokens)
                cleaned_episode.append(cleaned_sentence)

        return cleaned_episode

