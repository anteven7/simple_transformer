import torch
import math
class SimpleAttention(torch.nn.Module):
    def __init__(self, input_dim: int = 512, d_model: int = 128, heads: int = 4):

        super().__init__() 
        self.input_dim = input_dim
        self.d_model = d_model
        self.heads = heads
        self.q = torch.nn.Linear(input_dim, d_model)
        self.k = torch.nn.Linear(input_dim, d_model)
        self.v = torch.nn.Linear(input_dim, d_model)
        self.output = torch.nn.Linear(d_model, d_model)
        self.head_dim = d_model // heads

    def _dot_product(self, q, k , v):

        #their size is like (batch size, heads, seqlen, head_dim)
        #lets transpose k heads and head dim
        # we do this as we want to compute qk within the same "head" so we are looking for
        #(seqlen, head_dim) x (head_dim, seqlen) to this to be feasible

        dk = math.sqrt(k.size(-1))

        kt = k.transpose(-2, -1)
        
        scores = (q@kt)/dk # now we have (batch, heads, qseq, kseq)

        #to do - masking

        scores = torch.softmax(scores, dim = -1)
        attn = scores@v #(qseq, kseq)x(vseq, head_dim)

        return attn

    def attn(self, x):
        # x is ((batch_size, sequence_length, input_dim)): 
        print(f"x.size(): {x.size()}")
        batch_size, seq_len , input_dim = x.size()

        # batch_size : number of sequences processed concurrently in a single forward or backward pass.
        # sequence_length : total number of discrete elements, tokens, or time steps contained within a single  sequence.
        # d_model : dimensionality of the vector used to represent each individual token

        #projecting q, k v
        #afther that they will adquire the shape (batch_size, sequence_length, d_model) as the
        #mul is (seq_len, d_model) (d_model, d_model)

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        #lets reshape them into (batch_size, sequence_length, heads, head_dim) (d_model =  n_heads x head_dim )

        q = q.reshape(batch_size, seq_len, self.heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.heads, self.head_dim)

        # now, for computation as we want to compute all heads in pararell we have to transpose them to be 
        # like ((batch_size, self.heads, seq_len, self.head_dim))
        #we use this permutation so its easier for the gpu to multiply all the data within a single head (seqlen, headdim)
        # if we kept the sequence_length, heads, head_dim we would have to grab a piece of (heads, head_dim), and 
        # then go to the next matrix and grab the following.... etc. Visual example here:

        #(head x head dim)
        #        head_dim1 head_dim2 head_dim3
        # head 1 [x            x           x]
        # head 2 [x            x           x]
        # head 3 [x            x           x]
    
        #now if we want to compute all head 1, we would have to jump from every head1 row to head1 row within the 
        #tensor, which is not effective.
        
        q = q.permute(0, 2, 1, 3).contiguous() #permute alters the metadata and contiguous applies the change
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        #now we compute attnt

        attn = self._dot_product(q, k, v) # this 
        
        #now we revert the order so we can merge heads again

        attn = attn.permute(0, 2, 1, 3).contiguous()

        attn = attn.reshape(batch_size, seq_len, self.d_model)

        out = self.output(attn)

        return out
        
  
class FFLayer():
    pass        
class Decoder():
    pass


if __name__ == "__main__":
    pass