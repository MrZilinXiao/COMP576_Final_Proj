## COMP576 Final Project: Prompt Tuning for Efficient Text-based Emotion Classification
Team members: Zilin Xiao, Yuanyuan Xu, Ruidi Chang, Chunyuan Deng 

### Getting Started
1. Prepare text data from MELD dataset: 
```bash
mkdir data && cd data && git clone https://github.com/declare-lab/MELD && mv MELD/data/MELD/*_sent_emo.csv .
```

2. Install required packages: `pip install -r requirements.txt`

3. Choose one of the following example scripts to run experiments:
```bash
bash scripts/direct_roberta_large.sh
bash scripts/prompt_roberta_large.sh
```