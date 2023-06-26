# Unsupervised Topic Segmentation of Podcasts using a pre-trained Transformer model

### Prerequisite 
#### Install required packages
```bash
pip install -r requirements.txt
```
#### Fetch pre-trained sentence embeddings
```bash
wget -P data/features https://storage.googleapis.com/unsupervised-topic-segmentation-seminar/features/embeddings                                                                               ─╯
```
```
python -m evaluation.evaluate_topic_segmention
```
Output Path: `data/ouptut`

### Dataset
Brian Midei, M. M. (2018), ‘Neural text segmentation on podcast transcripts’.
URL: https://github.com/bmmidei/SliceCast/blob/master

### Acknowledgments
Solbiati, A., Heffernan, K., Damaskinos, G., Poddar, S., Modi, S. & Cali, J. (2021),
‘Unsupervised topic segmentation of meetings with bert embeddings’.
URL: https://arxiv.org/abs/2106.12978