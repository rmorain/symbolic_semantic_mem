# AUTOGENERATED! DO NOT EDIT! File to edit: 01_dataset_builder.ipynb (unless otherwise specified).

__all__ = ['DatasetBuilder']

# Cell
import random
from rake_nltk import Rake
from .database_proxy import WikiDatabase
import json
import importlib
import spacy
import tensorflow_hub as hub
from sklearn.neighbors import NearestNeighbors

# Cell
class DatasetBuilder():
    def __init__(self):
        self.rake = Rake()
        self.db = WikiDatabase()
        self.nlp = spacy.load('en_core_web_sm')
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
        self.encoder = hub.load(module_url)
        pass

    def build(self, ds, dataset_type='random'):
        "Build a database based a given dataset"
        if dataset_type == 'random':
            ds.map(self.random, batched=False)
        elif dataset_type == 'description':
            pass
        elif dataset_type == 'relevant':
            pass

    def keyword(self, x):
        ranked_phrases = self.get_ranked_phrases(x)
        return ranked_phrases[0]

    def get_ranked_phrases(self, x):
        self.rake.extract_keywords_from_text(x)
        return self.rake.get_ranked_phrases()

    #staticmethod
    def add_to_accepted(self, a_sentences, sentence):
        if len(a_sentences) > 2:
            a_sentences.pop(0)
        a_sentences.append(sentence)


    def get_entities_in_text(self, x, random_entities=False):
        accepted_sentence = []
        accepted_entities = []
        print(self.nlp)
        doc = self.nlp(x)
        for entity in doc.ents:
            if entity.label_ == 'CARDINAL':
                continue
            print(entity.text, entity.label_)
            result = self.db.get_entity_by_label(entity.text)
            if len(result) == 0:
                continue
            elif len(result) > 1:
                result = self.db.get_entities_by_label_extensive(entity.text)
                if len(accepted_sentence) == 0 or random_entities:
                    q_a_index = random.randint(0, len(result) - 1)
                else:
                    encoded_sentences = self.encoder(accepted_sentence)  # array of sentence vectors

                    proposed_sentences = []
                    for entity_w in result:
                        proposed_sentences.append(entity_w[2])
                    encoded_proposed = self.encoder(proposed_sentences)
                    neigh = NearestNeighbors(n_neighbors=1)
                    neigh.fit(encoded_proposed)
                    closest = neigh.kneighbors(encoded_sentences)
                    q_a_index = closest[1][0][0]
                self.add_to_accepted(accepted_sentence, result[q_a_index][2])
                accepted_entities.append(result[q_a_index])

            else:
                # print('Accepted:', result[0])
                self.add_to_accepted(accepted_sentence, result[0][2])
                accepted_entities.append(result[0])
        return accepted_entities

    def entity(self, ranked_phrases):
        "Queries the knowledge base to find the entity and it's relations"
        for phrase in ranked_phrases:
            entity = self.kba.get_entity(phrase)
            if entity is not None:
                return entity
        return entity
    def get_entity_properties_strings(self, entity_id):
        entity_properties_dict = {}
        for entity_property in self.db.get_entity_properties(entity_id):
            property_name, related_entity_label = self.db.get_property_string(entity_property[0], entity_property[1])
            entity_properties_dict[property_name] = related_entity_label
        return entity_properties_dict