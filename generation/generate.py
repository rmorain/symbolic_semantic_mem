import pprint

import spacy
from kirby.database_proxy import WikiDatabase
from kirby.run_params import RunParams
from transformers import GPT2Tokenizer, pipeline, set_seed


def convert_to_strings(entities):
    """
    Given a tuple of entities, return a list of strings
    """
    return [e.text for e in entities]


def get_entities(sequence):
    """
    Augment the data by adding an `entities` column to the dataset
    """
    entities = nlp(sequence).ents
    entity_strings = convert_to_strings(entities)
    return entity_strings


def get_knowledge_from_entities(entities):
    knowledge_list = []
    for entity in entities:
        knowledge = db.get_knowledge(entity)
        if knowledge:
            knowledge_list.append(knowledge)
    return knowledge_list


def filter_knowledge(knowledge):
    """
    Filter the knowledge
    """
    label = knowledge["label"]
    description = knowledge["description"]
    statement = f"{label} is a {description}. "
    return statement


def prepare_prompt(prompt, prompt_length, knowledge_length, prev_entities):
    entities = get_entities(prompt)
    knowledge_list = get_knowledge_from_entities(entities)
    statement = get_statement(knowledge_list)
    if not statement:
        entities = prev_entities  # If no valid entities found, keep prev entities
        knowledge_list = get_knowledge_from_entities(entities)
        statement = get_statement(knowledge_list)
    # Trim prompt tokens
    prompt_tokens = tokenizer(prompt)["input_ids"]
    knowledge_tokens = tokenizer(statement)["input_ids"]
    prompt = tokenizer.decode(prompt_tokens[-prompt_length:])
    knowledge = tokenizer.decode(knowledge_tokens[: knowledge_length - 2])
    knowledge = "(" + knowledge + ")"
    return knowledge + " " + prompt, entities


def get_statement(knowledge_list):
    for knowledge in knowledge_list:
        if "Wikimedia disambiguation page" in knowledge["description"]:
            continue
        return filter_knowledge(knowledge)
    return None


def generate_text(prompt, pipe, seq_length, prev_entities):
    """
    Given a prompt, generate some text
    """
    knowledge_prompt, entities = prepare_prompt(prompt, 15, 15, prev_entities)
    print(knowledge_prompt)
    generated_text = pipe(
        knowledge_prompt,
        max_length=seq_length,
        num_return_sequences=1,
        return_full_text=False,
        return_tensors=True,
    )
    return {
        "text": generated_text[0]["generated_text"],
        "knowledge_prompt": knowledge_prompt,
        "entities": entities,
    }


pipe = pipeline("text-generation", model="gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

set_seed(42)

nlp = spacy.load(
    "en_core_web_sm",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
)
rp = RunParams()
db = WikiDatabase(rp)

prompt = "Harry Potter raised his wand and said "
depth = 10
seq_length = 60
prompt_length = 30
pp = pprint.PrettyPrinter(indent=2)
log = []
entities = None

for i in range(depth):
    output = generate_text(prompt, pipe, seq_length, entities)
    generated_text = output["text"]
    prompt = generated_text
    entities = output["entities"]
    log.append(output)

pp.pprint(log)
