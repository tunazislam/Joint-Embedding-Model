# Joint-Embedding-Model

This repository contains code for [[Analysis of Twitter Users’ Lifestyle Choices using Joint Embedding Model]](https://ojs.aaai.org/index.php/ICWSM/article/view/18057), ICWSM 2021.

## Data:

1. Please download the 'data' folder from following link (on request):

https://drive.google.com/drive/folders/1_3TrcwVE2iWKL88fblDluEA5vJ3R0qvf?usp=sharing

2. 'data' folder should be kept inside the 'Joint-Embedding-Model/code' folder. 

for yoga: data/yoga_user_name_loc_des_mergetweets_yoga_1300_lb.csv file contains 6 columns: name, location, description, text, utype, umotivation


for keto: data/weakly_gt_mergetweets_keto_1300.csv


3. Train and Test data are inside data folder. 

data/train.txt

data/test.txt

data/train_keto.txt

data/test_keto.txt

### Computing Machine:

```
Supermicro SuperServer 7046GT-TR

Two Intel Xeon X5690 3.46GHz 6-core processors

48 GB of RAM

Linux operating system

Four NVIDIA GeForce GTX 1060 GPU cards for CUDA/OpenCL programming

```

### Software Packages and libraries:

```
python 3.6.6

PyTorch 1.1.0

jupiter notebook

pandas

gensim

nltk

nltk.tag

spacy

emoji

sklearn

statsmodels

scipy

matplotlib

numpy

preprocessor

transformers

```

## Run all codes from 'code' directory.

```
cd code

```

## Construct Yoga User Networks:

1) data/yoga_user_name_loc_des_mergetweets_yoga_1300_lb.csv file) (save in yoga_mergetweets_gt_1300.csv file containing 7 columns: name, location, description, text, utype, umotivation, uid) and @-mention network (save in data/yoga_graph_@mention.txt file)

```
get_at_mention_network_yoga_1300.ipynb

```
2) Create user network embeddings using Node2Vec and input graph for this is data/yoga_graph_@mention.txt


Please download Node2Vec from this link:

https://github.com/aditya-grover/node2vec
 
This will create following embedding:

data/userNetworkEmb.emb


## Run Models:

1) Run joint embedding model

```
BERT_longroberta_des+loc+Net+twt_utype.ipynb

BERT_longroberta_des+loc+Net+twt_umotivation.ipynb

```

2) Run description only baseline model.

```
baseline_BERT_description_utype.ipynb

baseline_BERT_description_umotivation.ipynb

```

3) Run location only baseline model. 
```
baseline_BERT_location_utype.ipynb

baseline_BERT_location_umotivation.ipynb

```

4) Run tweets only baseline model.

```
baseline_LongROBERTA_tweet_utype.ipynb

baseline_LongROBERTA_tweet_umotivation.ipynb

```

5) Run user network only baseline model. 

```
Net_utype.ipynb

Net_umotivation.ipynb

```

6) Run joint Description and Location (Des + Loc) model.

```
BERT_des+loc_utype.ipynb

BERT_des+loc_umotivation.ipynb

```

7) Run joint Description and Network (Des + Net) model.

```
BERT_des+Net_utype.ipynb

BERT_des+Net_umotivation.ipynb

```

8) Run joint Description, Location, Tweet (Des + Loc + Twt) model.

```
BERT_longroberta_des+loc+twt_utype.ipynb

BERT_longroberta_des+loc+twt_umotivation.ipynb

```

9) Run joint Description, Location, Network (Des + Loc + Net) model.

```
BERT_des+loc+Net_utype.ipynb

BERT_des+loc+Net_umotivation.ipynb

```

10) Run fine-tuned BERT model on Description (Description_BERT). 

```
baseline_BERT_finetuned_description_utype_preprocessed.ipynb

baseline_BERT_finetuned_description_umotivation_preprocessed.ipynb

```

11) Run fine-tuned BERT model on Location (Location_BERT).

```
baseline_BERT_finetuned_location_utype_preprocessed.ipynb

baseline_BERT_finetuned_location_umotivation_preprocessed.ipynb

```

12) Run fine-tuned BERT model on Tweets (Tweets_BERT). 

```
baseline_BERT_finetuned_tweet_utype_preprocessed_split.ipynb

baseline_BERT_finetuned_tweet_umotivation_preprocessed_split.ipynb

```

## For Keto diet:

### Construct Keto User Networks:

1) Get unique user id of 1300 keto users (input: data/weakly_gt_mergetweets_keto_1300.csv file) (save in data/keto_mergetweets_gt_1300.csv file containing 6 columns: name, location, description, text, utype_gt, uid) and create @-mention network (save in data/keto_graph_@mention.txt file): 
 

```
get_at_mention_network_keto_1300.ipynb

```

2) Create keto user network embeddings using Node2Vec and input graph for this is data/keto_graph_@mention.txt

Please download Node2Vec from this link:

https://github.com/aditya-grover/node2vec
 
This will create following embedding:

data/keto_userNetworkEmb_@mention.emb

### Run model:

```
keto_BERT_longroberta_des+loc+Net+twt_utype_new_usergraph.ipynb

```

## Citation:

If you find the paper useful in your work, please cite:

```
@inproceedings{islam2021analysis,
  title={Analysis of Twitter Users' Lifestyle Choices using Joint Embedding Model},
  author={Islam, Tunazzina and Goldwasser, Dan},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={15},
  pages={242--253},
  year={2021}
}

```


