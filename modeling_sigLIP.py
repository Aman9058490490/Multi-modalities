from typing import Optional,Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768, #size of the embedding vectore 
        intermediate_size=3072,#size of linear layer we used in feed forward network.
        num_hidden_layers=12, # number of layer are used in vision tranformer
        num_attention_heads=12,# number of head is used,like the number of tokens
        num_channels=3,# Number of Channels like R,G,B
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,#layer normalization eps
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens
class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__() #_init_ constructor which accept the config parameters
        self.config= config #Store the config inside the class
        self.embed_dim= config.hidden_size #this define that what is the dimension of embedding vector
        self.image_size = config.image_size
        self.patch_size= config.patch_size

        self.patch_embedding = nn.Conv2d( # patch_embedding are the convolutional layer which use to divide the image to the patches.
            in_channels=config.num_channels,
            out_channels=self.embed_dim,#which are equal to the embedding dimension of all patch.
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="Valid",#this indicates no padding is added
        )

        self.num_patches =(self.image_size// self.patch_size)**2 # basicaly thsi formula to divide the image in patches.now why the squr so cause (self.image_size x self.image_size)//(self.patch_Size x self.patch_Size)
        self.nm_positions=self.num_patches # it define that number of poastion are equal to the number of patches.
        self.positional_embedding = nn.Embedding(self.num_positions,self.embed_dim) # now on that it perform the position embedding.
        self.register_buffer(#this is model of pytorch which which store the or assigen the position to every patches and tokens. 
            "position_ids", 
             torch.arange(self.num_positions).expand((1, -1)) #it make the 1D tensor(array) which values range in btw 0 to n-1 where n is size of the tensor and expand(1,-1) means row is 1D and column is no change in previous dim.
             persistent=False, # means this buffer are not the part of model state dictionary.this buffer is usefull when we need to perform the temprory computation.
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
      _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
      # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
      # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
      # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
      patch_embeds = self.patch_embedding(pixel_values)  
      # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
      # where Num_Patches = Num_Patches_H * Num_Patches_W
      embeddings = patch_embeds.flatten(2)
      # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
      embeddings = embeddings.transpose(1, 2)
      # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
      embeddings = embeddings + self.position_embedding(self.position_ids)
      # [Batch_Size, Num_Patches, Embed_Dim]
      return embeddings
    
class SiglipAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim) this os formula which divide the similarty score,which we find to multiply the query vector and key vector.
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)# this is the weight vector of the query. so if we find the Query vectore we need multiply embedding vector and the weight vector(Wq)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)# similarily for others...
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim]
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights
    
class SiglipMLP(nn.modulde):
   def __init__(self,config):
      super().__init__()
      self.config = config
      self.fc1 = nn.Linear(config.hidden_size,config.intermediate_size) 
      self.fc2= nn.layer(config.intermediate_size,config.hidden_size)
    
   def forward(self,hidden_states: torch.Tensor)-> torch.Tensor:
       # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches,Intermediate_Size]
       hidden_states=self.fc1(hidden_states)
       #hidden_states: [Batch_size,Num_Patches,Intermediate_Size]
       hidden_states=nn.functional.gelu(hidden_states,approximate="tanh") # gelu(Gussian Error Linear Unit) is the activation function
       #which decide that to off the neuron or on the neuron.gelu are much flexibale towrds the input,means when we provide the input it pass to
    # next neuron after compress that input, which manage the fellow neurons.

class SigLipEncoderLayer(nn.Module): # nn.module yha pr pytorch ki base class ko inherit kr rha for neural networks.
  def __init__(self, config: SiglipVisionConfig):
      super().__init__()
      self.embed_dim=config.hidden_size
      self.self_attn = SiglipAttention(config)
      self.layer_norm1=nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps) # in that the layernorm1 and 2 for the layer normalization means layernorm1 perform when we add the result of multihead attention and the input which we provide the encoder.
      self.mlp=SiglipMLP(config)# after that normalization we apply MLP as per architecture after that we again use
      self.layer_norm2=nn.LayerNorm(self.embed_dim,eps=config.layer_norm_eps)# layernorm2 to normalise the input which we provide the MLP and the output of MLP.

  def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
    # residual: [Batch_Size, Num_Patches, Embed_Dim] batch size define how many images we take in one time.
       residual = hidden_states
       # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states = self.layer_norm1(hidden_states)
       # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states, _ = self.self_attn(hidden_states=hidden_states)
       # [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states = residual + hidden_states
       # residual: [Batch_Size, Num_Patches, Embed_Dim] 
       residual = hidden_states
       # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states = self.layer_norm2(hidden_states)
       # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states = self.mlp(hidden_states)
       # [Batch_Size, Num_Patches, Embed_Dim]
       hidden_states = residual + hidden_states
       
       return hidden_states
     
       