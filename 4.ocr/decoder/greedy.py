import torch

class GreedyDecoder:
    def __init__(self,vocabulary):
        self.vocabulary = vocabulary
        
    def decode(self,logits):
        # logits (B,T,C)
        predictions = torch.argmax(logits,dim=-1)
        
        texts = []
        for prediction in predictions:
            previous = None
            characters = []
            
            for idx in prediction.tolist():
                # remove blank 
                if idx == self.vocabulary.blank_idx:
                    previous = None
                    continue
                
                # remove duplicate
                if idx == previous:
                    continue
                
                characters.append(
                    self.vocabulary.idx_to_char[idx]
                )
                previous = idx
                
            texts.append(
                "".join(characters)
            )
        
        return texts
        