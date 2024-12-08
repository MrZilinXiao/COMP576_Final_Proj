from torch.utils.data import Dataset
import pandas as pd

emotion2id = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 
              'joy': 4, 'disgust': 5, 'anger': 6}

def utterance_filter(utterance):
    # filter """ and " with strip func, replace ’ with '
    return utterance.strip('"""').strip('"').replace('’', "'")
    

class MELDDataset(Dataset):
    # load from *_sent_emo.csv
    # Example: 
    # Sr No.,Utterance,Speaker,Emotion,Sentiment,Dialogue_ID,Utterance_ID,Season,Episode,StartTime,EndTime
    # 1,"Oh my God, he’s lost it. He’s totally lost it.",Phoebe,sadness,negative,0,0,4,7,"00:20:57,256","00:21:00,049"
    def __init__(self, data_root, tokenizer=None, split='train', prompt_dataset=False):
        super(MELDDataset, self).__init__()
        self.tokenizer = tokenizer
        self.data = pd.read_csv(f'{data_root}/{split}_sent_emo.csv')
        self.prompt_dataset = prompt_dataset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        utterance = utterance_filter(row['Utterance'])
        if self.prompt_dataset:  # prompt tuning needs special treatment
            return utterance, emotion2id[row['Emotion']]
        ret = {
            'label': emotion2id[row['Emotion']],
            **self.tokenizer(utterance, truncation=True)
        }
        # print(ret)
        # exit(0)
        return ret