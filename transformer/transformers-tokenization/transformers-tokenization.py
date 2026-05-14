import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        words = []
        for text in texts:
          for word in text.lower().split():
              if word in words:
                continue
              words.append(word) 
        sorted_words = sorted(words)

        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        for i in range(len(sorted_words)):
          self.word_to_id[sorted_words[i]] = i+4

        self.vocab_size = len(self.word_to_id)

        self.id_to_word = {value:key for key,value in self.word_to_id.items()}


        return self.word_to_id, self.id_to_word
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        text = text.lower()
        token = text.split()
        enc = []

        for i in range(len(token)):
          if token[i] in self.word_to_id.keys():
            enc.append(self.word_to_id.get(token[i]))
          else:
            enc.append(self.word_to_id.get(self.unk_token))


        return enc

    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        dec = []
        for id in ids:
            dec.append(self.id_to_word.get(id, self.unk_token))
        dec = " ".join(dec)
        return dec
