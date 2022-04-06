# MLMC

Code for ACL 2021 Paper [Argument Pair Extraction via Attention-guided Multi-Layer Multi-Cross Encoding](https://aclanthology.org/2021.acl-long.496.pdf).

## Packages and Environment

``` 
python                    3.7.9 
torch                     1.7.1 
termcolor                 1.1.0 
transformers              4.1.1 
tqdm                      4.55.1
```

## Data

Data is the second version for dataset described in this [paper](https://aclanthology.org/2020.emnlp-main.569.pdf). \
All train/dev/test data is split beforehand in JSON format. \
Training data is uploaded in zip format because of size limit.

## Usage

```
cd model/CrossModel 
python main.py (with default parameters)
```

## Citation
```
@inproceedings{cheng2021argument,
  title={Argument Pair Extraction via Attention-guided Multi-Layer Multi-Cross Encoding},
  author={Cheng, Liying and Wu, Tianyu and Bing, Lidong and Si, Luo},
  booktitle={Proceedings of ACL-IJCNLP},
  year={2021}
}
```
