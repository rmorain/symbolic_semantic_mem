import concurrent.futures
import random

import spacy
import torch
import torch.nn.functional as F
from kirby.database_proxy import WikiDatabase
from kirby.knowledge_model import KnowledgeModel
from kirby.run_params import RunParams
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed


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
        knowledge = db.get_associated_entities(entity)
        if knowledge:
            knowledge_list.append(knowledge)
    return knowledge_list


def format_associated_entities(knowledge):
    # Randomly pick associated entity
    associated_entities = knowledge["associated_entities"]
    if len(associated_entities) > 0:
        rand_index = random.randint(0, len(associated_entities) - 1)
        rand_entity = associated_entities[rand_index]
        relations = list(rand_entity.items())
        x = relations[0][1] + "'s " + relations[2][0] + " is " + relations[2][1]
        return x
    else:
        return filter_knowledge(knowledge)


def filter_knowledge(knowledge, associated_entities=False):
    """
    Filter the knowledge
    """
    if associated_entities:
        return format_associated_entities(knowledge)
    label = knowledge["label"]
    description = knowledge["description"]
    statement = f"{label} is a {description} "
    return statement


def prepare_prompt(prompt, prompt_length, knowledge_length, prev_entities):
    entities = get_entities(prompt)
    knowledge_list = get_knowledge_from_entities(entities)
    statement_list = get_statement(knowledge_list)
    if not statement_list:
        entities = prev_entities  # If no valid entities found, keep prev entities
        knowledge_list = get_knowledge_from_entities(entities)
        statement_list = get_statement(knowledge_list)
    # Trim prompt tokens
    prompt_tokens = tokenizer(prompt)["input_ids"]
    prompt = tokenizer.decode(prompt_tokens[-prompt_length:])
    return statement_list, prompt, entities


def get_statement(knowledge_list):
    statement_list = []
    for knowledge in knowledge_list:
        if "Wikimedia disambiguation page" in knowledge["description"]:
            continue
        if knowledge["description"] == "":
            continue

        statement = filter_knowledge(knowledge, associated_entities=True)
        statement_list.append(statement)
    return statement_list


class Node:
    def __init__(self, text, prompt, knowledge, entities):
        self.text = text
        self.entities = entities
        self.prompt = prompt
        self.knowledge = knowledge
        self.children = []

    def __repr__(self, level=0):
        ret = "\t" * level + repr(self.text) + f" (Entities: {self.entities})" + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


def generate_text(prompt, pipe, seq_length, prev_entities, knowledge, depth):
    """
    Given a prompt, generate some text
    """
    if depth == 0:
        return None
    generated_text = pipe(
        prompt,
        prefix=knowledge,
        max_length=seq_length,
        num_return_sequences=1,
        return_full_text=False,
        return_tensors=False,
    )
    node = Node(
        generated_text[0]["generated_text"],
        prompt,
        knowledge,
        prev_entities,
    )
    knowledge, prompt, entities = prepare_prompt(node.text, 15, 15, prev_entities)

    for k in knowledge:
        child_node = generate_text(prompt, pipe, seq_length, entities, k, depth - 1)
        if child_node:
            node.children.append(child_node)

    return node


def print_text(node, indent):
    if node:
        print(f"{'   ':<{indent}} {node.text}")
        [print_text(child, indent + 1) for child in node.children]
    return None


def score(lines):
    lines = [tokenizer.eos_token + line for line in lines]

    tok_res = tokenizer.batch_encode_plus(lines, return_tensors="pt", padding="longest")
    input_ids = tok_res["input_ids"]
    attention_mask = tok_res["attention_mask"]
    lines_len = torch.sum(tok_res["attention_mask"], dim=1)

    logits = []
    for i, ids in enumerate(input_ids):
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask[i],
            labels=ids,
        )
        logits.append(outputs.logits)
    logits = torch.stack(logits)

    for line_ind in range(len(lines)):
        line_log_prob = 0.0
        for token_ind in range(lines_len[line_ind] - 1):
            token_prob = F.softmax(logits[line_ind, token_ind], dim=0)
            token_id = input_ids[line_ind, token_ind + 1]
            line_log_prob += torch.log(token_prob[token_id])
        lines[line_ind] = (lines[line_ind][13:], line_log_prob.item())
    lines.sort(key=lambda x: x[1])
    return lines


def build_sequences(node):
    if not node.children:
        return [node.text]
    sequences = []
    for child in node.children:
        sequences += build_sequences(child)
    for i in range(len(sequences)):
        sequences[i] = (
            f"({node.entities}) " + f"({node.knowledge}) " + node.text + sequences[i]
        )

    return sequences


rp = RunParams(
    pretrained=True,
)
model = GPT2LMHeadModel.from_pretrained("gpt2")
pipe = pipeline("text-generation", model="gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

set_seed(42)

nlp = spacy.load(
    "en_core_web_sm",
    disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"],
)
db = WikiDatabase(rp)

prompts = [
    "Barack Obama was the 44th president of the US. ",
    "Harry Potter raised his wand and said ",
    "I heard Steve Jobs invented the iPhone. ",
]
depth = 3
seq_length = 90
prompt_length = 30
f = open("results.txt", "w")

for prompt in prompts:
    knowledge, prompt, entities = prepare_prompt(prompt, 15, 15, None)
    print(f"Prompt: {prompt}", file=f)
    root_node = generate_text(prompt, pipe, seq_length, entities, knowledge[0], depth)
    sequences = build_sequences(root_node)
    sequences = score(sequences)
    # Print median sequence
    print("Min knowledge sequence", file=f)
    print(sequences[0][0], file=f)
    print("Median knowledge sequence", file=f)
    print(sequences[len(sequences) // 2][0], file=f)
    print("Max knowledge sequence", file=f)
    print(sequences[-1][0], file=f)

    # Prompt30
    f.write("Baseline generation")
    output = ""
    for i in range(depth):
        generated_text = pipe(
            prompt,
            max_length=seq_length,
            num_return_sequences=1,
            return_full_text=False,
            return_tensors=True,
        )
        prompt_tokens = tokenizer(generated_text[0]["generated_text"])
        prompt = tokenizer.decode(prompt_tokens["input_ids"])[-seq_length:]
        output += generated_text[0]["generated_text"]
    print(output, file=f)
f.close()

# Open ended text generation
# prompt = "I heard that Steve Jobs invented the iPhone "
# print("Open ended text generation")
# generated_text = pipe(
# prompt,
# max_length=seq_length * depth,
# num_return_sequences=1,
# return_full_text=False,
# return_tensors=True,
# )
# print(generated_text[0]["generated_text"])
