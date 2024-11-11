from typing import List

class EncoderDecoder:
    def __init__(self, vocab:List[str]):
        # Letter to number
        self.encoder_dict = {}
        # Number to letter
        self.decoder_dict = {}
        # Generate encoder and decoder dicts
        for i, char in enumerate(vocab):
            self.encoder_dict[char] = i
            self.decoder_dict[i] = char
        

    def encode(self, data: str) -> List:
        encoded_data = []
        for d in data:
            encoded_data.append(self.encoder_dict[d])
        
        return encoded_data

    def decode(self, data:List) -> str:
        decoded_data = ""
        for d in data:
            decoded_data += self.decoder_dict[d]

        return decoded_data