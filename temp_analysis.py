#%% 
import torch 
import torch
input_ids_temp = torch.load('temp/input_ids.pt')
labels_temp = torch.load('temp/labels.pt')

print(input_ids_temp)
print(labels_temp)

eos_positions = (input_ids_temp == 2 ).nonzero()
print(eos_positions)
# there is masking with ~ and there is broadcasting indexing
mask_eos = (input_ids_temp == 2)
print(input_ids_temp[~mask_eos].view(input_ids_temp.shape[0], -1  ))
print(labels_temp[~mask_eos].view(labels_temp.shape[0], -1  ))

# %%
