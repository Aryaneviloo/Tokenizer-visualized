import re

class Tokenizer:
    def __init__(self, vocab_size=275):
        self.vocab_size=vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.split_pattern = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[^?\s\w\d]|[a-zA-Z]+|[0-9]+|(?:\s*[\r\n]+|\s+)"""
        )
    def _pre_split(self, text):
        """
        Takes raw text, applies the production regex pattern, 
        and converts each isolated chunk into a list of byte integers.
        """
        text_chunks=self.split_pattern.findall(text)
        byte_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks] 
        return byte_chunks
    def _get_stats(self, chunks):
        """
        Counts the frequencies of consecutive integer pairs, 
        ensuring boundaries between pre-split chunks are never crossed.
        """
        counts = {}
        for chunk in chunks:
            # We iterate through the current individual chunk exactly as before
            for pair in zip(chunk, chunk[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts  
    def _merge(self, chunks, pair, idx):
        new_chunks = []
        for chunk in chunks:
            new_chunk=[]
            i=0
            while i < len(chunk):
                if i <len(chunk)-1 and chunk[i] == pair[0] and chunk[i+1] == pair[1]:
                    new_chunk.append(idx)
                    i += 2
                else: 
                    new_chunk.append(chunk[i])
                    i += 1

            new_chunks.append(new_chunk)

        return new_chunks        
    
    def train(self, text):
        num_merges = self.vocab_size - 256
        chunks = self._pre_split(text)
        for i in range(num_merges):
            stats = self._get_stats(chunks)
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            chunks = self._merge(chunks, top_pair, idx)
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

#Reverses a list of token IDs back into a human-readable string by reconstructing their raw byte lineages.
    def decode(self, ids):
        raw_bytes = b"".join(self.vocab[idx] for idx in ids)
        return raw_bytes.decode("utf-8", errors= "replace")



#Converts a new raw string into a compressed list of token ids using the learned merge rules from training.
   
    def encode(self, text):
        chunks = self._pre_split(text)
        while len(chunks) >= 1:
            stats = self._get_stats(chunks)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            chunks= self._merge(chunks, pair, idx)

        flatten_ids = []
        for chunk in chunks:
            flatten_ids.extend(chunk)

        return flatten_ids
    
    def get_visual_tokens(self, ids):
        """
        Converts a list of integer token IDs back into individual 
        string fragments specifically for frontend visualization blocks.
        """
        return [self.vocab[idx].decode("utf-8", errors="replace") for idx in ids]
    
    def save(self, file_prefix):
        """
        Serializes the learned merges and vocabulary to disk 
        so it can be reloaded without retraining.
        """
        model_file = f"{file_prefix}.model"
        with open(model_file, "w", encoding="utf-8") as f:
            f.write("--- Custom BPE Merges Registry ---\n")
            for (p0, p1), idx in self.merges.items():
                f.write(f"{p0} {p1} -> {idx}\n")