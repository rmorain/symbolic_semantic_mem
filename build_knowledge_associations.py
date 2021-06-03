from datasets import Dataset 
from kirby.database_proxy import get_knowledge

def add_knowledge(example):
    """
    Adds a `knowledge` column to the data point
    Knowledge is a list of dictionaries.

    Example:
        [
            {
                'description':'A very good description.', 
                'association': 'More associations',
                ...
            }
        ]
    """
    knowledge_list = []
    for entity in example['entities']:
         

# Load dataset
split = 'train'
save_location = 'data/augmented_datasets/entities/' + split + '/'
ds = Dataset.load_from_disk(save_location)

# Augment data
ds = ds.map(add_knowledge)
