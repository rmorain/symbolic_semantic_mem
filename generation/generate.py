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
    statement = f"{label} is a {description} "
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
    # knowledge_tokens = tokenizer(statement)["input_ids"]
    prompt = tokenizer.decode(prompt_tokens[-prompt_length:])
    knowledge = statement
    # knowledge = tokenizer.decode(knowledge_tokens[: knowledge_length - 2])
    return knowledge, prompt, entities


def get_statement(knowledge_list):
    for knowledge in knowledge_list:
        if "Wikimedia disambiguation page" in knowledge["description"]:
            continue
        if knowledge["description"] == "":
            continue

        return filter_knowledge(knowledge)
    return None


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


def generate_text(prompt, pipe, seq_length, prev_entities, depth):
    """
    Given a prompt, generate some text
    """
    if depth == 0:
        return None
    knowledge, prompt, entities = prepare_prompt(prompt, 15, 15, prev_entities)
    generated_text = pipe(
        prompt,
        prefix=knowledge,
        max_length=seq_length,
        num_return_sequences=1,
        return_full_text=False,
        return_tensors=True,
    )
    node = Node(
        generated_text[0]["generated_text"],
        prompt,
        knowledge,
        entities,
    )
    for entity in entities:
        child_node = generate_text(
            node.text, pipe, seq_length, node.entities, depth - 1
        )
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

    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
    )
    loss, logits = outputs[:2]

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


__import__("pudb").set_trace()
rp = RunParams(
    pretrained=True,
)
model = KnowledgeModel(rp)
pipe = pipeline("text-generation", model=model.model, tokenizer="gpt2")
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
seq_length = 120
prompt_length = 30
entities = None
f = open("results.txt", "w")

for prompt in prompts:
    print(f"Prompt: {prompt}", file=f)
    root_node = generate_text(prompt, pipe, seq_length, entities, depth)
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
        prompt_tokens = generated_text[0]["generated_token_ids"]
        prompt = tokenizer.decode(prompt_tokens)[-seq_length:]
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
