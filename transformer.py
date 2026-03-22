import torch
import math

class SimpleAttention(torch.nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 4):

        super().__init__() 
        self.d_model = d_model
        self.heads = heads
        self.q = torch.nn.Linear(d_model, d_model)
        self.k = torch.nn.Linear(d_model, d_model)
        self.v = torch.nn.Linear(d_model, d_model)
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

        #ahora el masking es vital para decoder-only ya que evitamos que la atención tenga información sobre futuros
        #tokens y atienda solamente a tokens previos (se ponen a -inf el triangulo superior de la matriz)
        mask = torch.tril(torch.ones(scores.size(-2), scores.size(-1), device=scores.device))
        #tril genera matriz triangular inferior de dimenisones (qseq y kseq)

        scores = scores.masked_fill(mask == 0, float('-inf'))
        #masked_fill aplica la máscara (cuando mask == 0 lo pone como -inf)
        #el vector scores es 4D mientras q mask es 2D. Esto lo soluciona masked_fill automaticamente con el broadcasting
        #de pytorch (rellena con 1s quedando (1, 1, seq_len, seq_len))


        scores = torch.softmax(scores, dim = -1)
        attn = scores@v #(qseq, kseq)x(vseq, head_dim)

        return attn

    def forward(self, x):
        # x is ((batch_size, sequence_length, d_model)): 
        print(f"x.size(): {x.size()}")
        batch_size, seq_len , d_model = x.size()

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
        # out is a tensor (batch size, seq_len, d_model)
        return out

        
class FFLayer(torch.nn.Module):

    def __init__(self, d_model, dropout=0.1):

        super().__init__()

        self.ff = nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4), #usually in transformers you expand d_model x 4 before turning it into d_model again 
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model * 4, d_model)
        )
        
    def forward(self, x):

        return self.ff(x)

class Decoder(torch.nn.Module):
    
    def __init__(self, d_model, n_heads, dropout):

        """
        we follow the traditional implementation of the decoder
        here is super important to understand LayerNorm and residual connections between the modules
        the basic decoder module is attn + ffw blocks
        
        las capas residuales (add) aseguran que no vamos a tener problemas de vanishing gradients al
        aplicar backpropagation. si el gradiente se va a 0 seguiremos sumando la identidad permitiendo al gradiente
        continuar (teoría de las conexiones residuales)
        
        se normaliza para mantener estos valores entre -1 y 1 con media en 0 y así solventar
        el problema causado por sumar en las conexiones residuales
    
        """
        super().__init__()
        self.attn = SimpleAttention(d_model, n_heads)
        self.ffw = FFLayer(d_model, dropout)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x):

        norm = self.norm1(x)
        attn = self.attn(norm)
        add = attn + x 
        
        
        norm = self.norm2(add)
        ffw = self.ffw(norm)
        add = add + ffw
        

        return add


if __name__ == "__main__":
    pass