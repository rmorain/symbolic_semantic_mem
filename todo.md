# Todo

1. Build random dataset
    - Iterate over each text input x
    - For each x, query and save a random association of the key entity
    - Train on random dataset
    
2. Build description dataset
    - Iterate over each text input x
    - For each x, query and save the description of key entity
    - Train on description dataset
    
3. Build relevant association dataset
    - Iterate over each text input x
    - For each x, retrieve all the associations of the key entity
    - Score them by relevance to the original input x
    - Save the top p associations to the dataset
    - Train on relevant association dataset