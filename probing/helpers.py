import os

import numpy as np
import openai
import torch
from torch.nn import functional as F
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    RobertaForMaskedLM, 
    RobertaTokenizer, 
    T5ForConditionalGeneration,
    T5Tokenizer
)

# import prompting

# Define path to attribute lists
ATTRIBUTES_PATH = os.path.abspath("../data/attributes/{}.txt")

# Define path to variables
VARIABLES_PATH = os.path.abspath("../data/pairs/{}.txt")

# Define path to continuation probabilities
PROBS_PATH = os.path.abspath("probs/")
if not os.path.exists(PROBS_PATH):
    os.makedirs(PROBS_PATH)  # Create folder if it does not exist

# Define model groups
GPT2_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
ROBERTA_MODELS = ["roberta-base", "roberta-large"]
T5_MODELS = ["t5-small", "t5-base", "t5-large", "t5-3b"]

# Define OpenAI names
OPENAI_NAMES = {
    "davinci": "gpt3-davinci",
    "gpt-4-0613": "gpt4",
    "text-davinci-003": "gpt3"
}

def load_model(model_name):
    if model_name in GPT2_MODELS:
        return GPT2LMHeadModel.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaForMaskedLM.from_pretrained(
            model_name
        )
    elif model_name in T5_MODELS:
        return T5ForConditionalGeneration.from_pretrained(
            model_name 
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")


# Function to load tokenizer
def load_tokenizer(model_name):
    if model_name in GPT2_MODELS:
        return GPT2Tokenizer.from_pretrained(
            model_name 
        )
    elif model_name in ROBERTA_MODELS:
        return RobertaTokenizer.from_pretrained(
            model_name 
        )
    elif model_name in T5_MODELS:
        return T5Tokenizer.from_pretrained(
            model_name 
        )
    else:
        raise ValueError(f"Model {model_name} not supported.")




# used for free predictions (without pre-defined options)
def get_top_predictions(prompt, model, model_name, tok, device, labels, top_k=5):
    input_ids = torch.tensor([tok.encode(prompt)])
    input_ids = input_ids.to(device)

    probs = compute_probs(
        model, 
        model_name, 
        input_ids, 
        labels
    )

    top_probs, top_indices = torch.topk(probs, top_k)
    top_tokens = [tok.convert_ids_to_tokens(idx.item()) for idx in top_indices]
    clean_tokens = [token.replace('▁', '').replace('##', '').replace('Ġ', '') for token in top_tokens]
    top_probs = top_probs.tolist()
    return list(zip(clean_tokens, top_probs))



def compute_probs(model, model_name, input_ids, labels):
    if model_name in GPT2_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-1]
    elif model_name in ROBERTA_MODELS:
        output = model(input_ids=input_ids)
        probs = F.softmax(output.logits, dim=-1)[0][-2]
    elif model_name in T5_MODELS:
        output = model(input_ids=input_ids, labels=labels)
        probs = F.softmax(output.logits, dim=-1)[0][-1] 
    else:
        raise ValueError(f"Model {model_name} not supported.")
    return probs



def get_attribute_log_probs(prompt, attributes, model, model_name, tok, device, labels):
    """
    Calculates the log-probability sum for each attribute's tokens given a prompt.

    Parameters:
        prompt (str): The input text prompt with a masked token.
        attributes (list): List of attribute words/phrases to calculate probabilities for.
        model (torch.nn.Module): The language model.
        model_name (str): Name of the model (used in some model-specific handling).
        tok (transformers.Tokenizer): The tokenizer associated with the model.
        device (str): Device to run computations on (e.g., "cpu" or "cuda").
        labels (torch.Tensor or None): Label tokens if needed for specific models.

    Returns:
        list: Log-probability sums for each attribute.
    """
    input_ids = torch.tensor([tok.encode(prompt)]).to(device)

    probs = compute_probs(
        model, 
        model_name, 
        input_ids, 
        labels
    )
    
    # Compute log-probability sum for each attribute
    log_probs_attribute = []
    for a in attributes:
        # Encode attribute and calculate log probabilities for each token
        token_ids = tok.encode(a, add_special_tokens=False)
        token_log_probs = [np.log(probs[token_id].item()) for token_id in token_ids]
        
        # Calculate the sum of log probabilities
        log_prob_sum = np.sum(token_log_probs)
        log_probs_attribute.append(log_prob_sum)
        
    return log_probs_attribute

#normal probs (not log)
def get_attribute_probs(prompt, attributes, model, model_name, tok, device, labels):
    input_ids = torch.tensor([tok.encode(prompt)])
    input_ids = input_ids.to(device)

    probs = compute_probs(
        model, 
        model_name, 
        input_ids, 
        labels
    )

    probs_attribute = [
        probs[tok.convert_tokens_to_ids(a)].item() for a in attributes
    ]
    return probs_attribute