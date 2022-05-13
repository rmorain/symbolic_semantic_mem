from copy import deepcopy

import pandas as pd
from transformers import GPT2Tokenizer

from attention_entity_selection import process_data
from build_knowledge_associations import process_knowledge
from max_attention import process_max_attention
from median_attention import process_median_attention
from min_attention import process_min_attention

DEBUG = False
PREFIX = ""

# Generate attention scores
DATA_PATH = PREFIX + "data/augmented_datasets/pickle/entities_valid.pkl"
df = pd.read_pickle(DATA_PATH)
SAVE_PATH = PREFIX + "data/augmented_datasets/pickle/sorted_attentions_valid_sealed.pkl"
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
main_df = process_data(df, SAVE_PATH, tokenizer, debug=DEBUG)
main_df = pd.read_pickle(SAVE_PATH)

# Process knowledge
DATA_PATH = PREFIX + "data/augmented_datasets/pickle/sorted_attentions_valid_sealed.pkl"
SAVE_PATH = PREFIX + "data/augmented_datasets/pickle/augmented_train_sealed.pkl"
main_df = process_knowledge(DATA_PATH, SAVE_PATH, debug=DEBUG)

# Max attention
SAVE_PATH = PREFIX + "data/augmented_datasets/pickle/max_attention_valid_sealed.pkl"
max_df = process_max_attention(deepcopy(main_df), SAVE_PATH, debug=DEBUG)

# Median attention
SAVE_PATH = PREFIX + "data/augmented_datasets/pickle/median_attention_valid_sealed.pkl"
med_df = process_median_attention(deepcopy(main_df), SAVE_PATH, debug=DEBUG)

# Min attention
SAVE_PATH = PREFIX + "data/augmented_datasets/pickle/min_attention_valid_sealed.pkl"
min_df = process_min_attention(deepcopy(main_df), SAVE_PATH, debug=DEBUG)
