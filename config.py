import torch as T


device = T.device("cuda" if T.cuda.is_available() else "cpu")

data_path = "data/podcast"
output_path = 'data/output/eval_output.csv'
bucket_name = "unsupervised-topic-segmentation-seminar"
source_features_path = "data/features/embeddings"
destination_feature_path = "features/"
