# Learning Intents behind Interactions with Knowledge Graph for Recommendation

This is our PyTorch implementation for the paper:

> Xiang Wang, Tinglin Huang, Dingxian Wang, Yancheng Yuan, Zhenguang Liu, Xiangnan He and Tat-Seng Chua (2021). Learning Intents behind Interactions with Knowledge Graph for Recommendation.  [Paper in arXiv](https://arxiv.org/abs/2102.07057). In WWW'2021, Ljubljana, Slovenia, April 19-23, 2021.

## Introduction

Knowledge Graph-based Intent Network (KGIN) is a recommendation framework, which consists of three components: (1)user Intent modeling, (2)relational path-aware aggregation, (3)indepedence modeling.

## Citation 

If you want to use our codes and datasets in your research, please cite:

```
@inproceedings{KGIN2020,
  author    = {Xiang Wang and
              Tinglin Huang and 
              Dingxian Wang and
              Yancheng Yuan and
              Zhenguang Liu and
              Xiangnan He and
              Tat{-}Seng Chua},
  title     = {Learning Intents behind Interactions with Knowledge Graph for Recommendation},
  booktitle = {{WWW}},
  pages     = {878-887},
  year      = {2021}
}
```

## Environment Requirement

The code has been tested running under Python 3.6.5. The required packages are as follows:

- pytorch == 1.5.0
- numpy == 1.15.4
- scipy == 1.1.0
- sklearn == 0.20.0
- torch_scatter == 2.0.5
- networkx == 2.5

## Reproducibility & Example to Run the Codes

To demonstrate the reproducibility of the best performance reported in our paper and faciliate researchers to track whether the model status is consistent with ours, we provide the best parameter settings (might be different for the custormized datasets) in the scripts, and provide [the log for our trainings](https://github.com/huangtinglin/Knowledge_Graph_based_Intent_Network/tree/main/training_log).

The instruction of commands has been clearly stated in the codes (see the parser function in utils/parser.py). 

- Last-fm dataset

```
python main.py --dataset last-fm --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3
```

- Amazon-book dataset

```
python main.py --dataset amazon-book --dim 64 --lr 0.0001 --sim_regularity 0.00001 --batch_size 1024 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3
```

- Alibaba-iFashion dataset

```
python main.py --dataset alibaba-fashion --dim 64 --lr 0.0001 --sim_regularity 0.0001 --batch_size 1024 --node_dropout True --node_dropout_rate 0.5 --mess_dropout True --mess_dropout_rate 0.1 --gpu_id 0 --context_hops 3
```

Important argument:

- `sim_regularity`
  - It indicates the weight to control the independence loss.
  - 1e-4(by default), which uses 0.0001 to control the strengths of  correlation. 

## Dataset

We provide three processed datasets: Amazon-book, Last-FM, and Alibaba-iFashion.

- You can find the full version of recommendation datasets via [Amazon-book](http://jmcauley.ucsd.edu/data/amazon), [Last-FM](http://www.cp.jku.at/datasets/LFM-1b/), and [Alibaba-iFashion](https://github.com/wenyuer/POG).
- We follow [KB4Rec](https://github.com/RUCDM/KB4Rec) to preprocess Amazon-book and Last-FM datasets, mapping items into Freebase entities via title matching if there is a mapping available.

|                       |               | Amazon-book |   Last-FM | Alibaba-ifashion |
| :-------------------: | :------------ | ----------: | --------: | ---------------: |
| User-Item Interaction | #Users        |      70,679 |    23,566 |          114,737 |
|                       | #Items        |      24,915 |    48,123 |           30,040 |
|                       | #Interactions |     847,733 | 3,034,796 |        1,781,093 |
|    Knowledge Graph    | #Entities     |      88,572 |    58,266 |           59,156 |
|                       | #Relations    |          39 |         9 |               51 |
|                       | #Triplets     |   2,557,746 |   464,567 |          279,155 |

- `train.txt`
  - Train file.
  - Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
- `test.txt`
  - Test file (positive instances).
  - Each line is a user with her/his positive interactions with items: (`userID` and `a list of itemID`).
  - Note that here we treat all unobserved interactions as the negative instances when reporting performance.
- `user_list.txt`
  - User file.
  - Each line is a triplet (`org_id`, `remap_id`) for one user, where `org_id` and `remap_id` represent the ID of such user in the original and our datasets, respectively.
- `item_list.txt`
  - Item file.
  - Each line is a triplet (`org_id`, `remap_id`, `freebase_id`) for one item, where `org_id`, `remap_id`, and `freebase_id` represent the ID of such item in the original, our datasets, and freebase, respectively.
- `entity_list.txt`
  - Entity file.
  - Each line is a triplet (`freebase_id`, `remap_id`) for one entity in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such entity in freebase and our datasets, respectively.
- `relation_list.txt`
  - Relation file.
  - Each line is a triplet (`freebase_id`, `remap_id`) for one relation in knowledge graph, where `freebase_id` and `remap_id` represent the ID of such relation in freebase and our datasets, respectively.

## Acknowledgement

Any scientific publications that use our datasets should cite the following paper as the reference:

```
@inproceedings{KGIN2020,
  author    = {Xiang Wang and
              Tinglin Huang and 
              Dingxian Wang and
              Yancheng Yuan and
              Zhenguang Liu and
              Xiangnan He and
              Tat{-}Seng Chua},
  title     = {Learning Intents behind Interactions with Knowledge Graph for Recommendation},
  booktitle = {{WWW}},
  pages     = {878-887},
  year      = {2021}
}
```

Nobody guarantees the correctness of the data, its suitability for any particular purpose, or the validity of results based on the use of the data set. The data set may be used for any research purposes under the following conditions:

- The user must acknowledge the use of the data set in publications resulting from the use of the data set.
- The user may not redistribute the data without separate permission.
- The user may not try to deanonymise the data.
- The user may not use this information for any commercial or revenue-bearing purposes without first obtaining permission from us.