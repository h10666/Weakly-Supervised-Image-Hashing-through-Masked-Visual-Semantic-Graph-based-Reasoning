### MGRN 
### Official pytorch implementation of the paper: " Weakly-Supervised Image Hashing through Masked Visual-Semantic Graph-based Reasoning", accepted by ACM Multimedia 2020


### Description
This paper proposes a novel Masked visual-semantic Graph-based Reasoning Network (MGRN) to learn joint visual-semantic representations for image hashing.

### Dataset
We evaluate the retrieval performance on two widely used datasets, including MIRFlick25k and NUS-WIDE. The experimental settings are as follows.

MIRFlickr25k:
This dataset can be download at https://press.liacs.nl/mirflickr/.
The split of the training set, query set and database set can be found in the path at 'dataset/mirflickr25k'


NUS-WIDE:
This dataset can be download at https://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
The saplit of the training set, query set and database set can be found in the path at 'dataset/nus'

### Testing
run the commond:
python -u eval_hash_gat_mir.py
python -u eval_hash_gat_nus.py

### Experimental Results

The mAP@all results for the MIRFlickr25k dataset         
8bits | 16bits | 32bits | 48bits | 64bits 
70.31 | 70.77  | 71.12  | 71.25  | 71.45

The mAP@all results for the NUS-WIDE dataset
8bits | 16bits | 32bits | 48bits | 64bits 
60.60 | 62.08  | 63.91  | 64.05  | 64.13  

完整代码 https://pan.baidu.com/s/1VO0sQKBdhsfb25SdIw18Rw?pwd=945i