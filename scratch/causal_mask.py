import torch
# if only "normal" attention layer implements causal mask
query_length = 8 
key_length = 8
max_positions = 8
knowledge_buffer = 8
bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ) 
knowledge_mask = torch.ones((knowledge_buffer, knowledge_buffer), dtype=torch.uint8).view(1, 1, knowledge_buffer, knowledge_buffer).bool()
print(knowledge_mask)
masked_bias = torch.tensor(-1e4)
print('bias', bias)
attn_weights = torch.tensor((), dtype=torch.float32)
# Needs to be the same size as causal_mask   
attn_weights = attn_weights.new_ones(size=(1, 12, 8, 16))
#print('attn_weights', attn_weights.shape)
causal_mask = bias[:, :, key_length - query_length : key_length, :key_length].bool()
causal_mask = torch.cat((causal_mask, knowledge_mask), 3)
print(causal_mask)
attn_weights = torch.where(causal_mask, attn_weights, masked_bias.to(attn_weights.dtype))
#print('attn_weights', attn_weights)
print('attn_weights', attn_weights.shape)
