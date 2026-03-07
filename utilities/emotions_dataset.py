import torch
from torch.utils.data import Dataset


class EmotionsDataset(Dataset):

    def __init__(self, content, labels, tokenizer, max_length):
        self.content = content
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        tweet = str(self.content[idx])
        label = self.labels[idx]

        encoded = self.tokenizer(tweet,
                                    padding=False, # dynamic padding with datacollector
                                    truncation=True,
                                    max_length=self.max_length)

        o = dict()
        o['input_ids'] = encoded['input_ids']
        o['attention_mask'] = encoded['attention_mask']
        o['labels']= torch.tensor(label, dtype=torch.long)
        return o

    def __len__(self):

        return len(self.content)
