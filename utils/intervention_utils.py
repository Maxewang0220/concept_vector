import torch
from baukit import TraceDict
from responses import target

from model_utils import get_module
from bitsandbytes import functional as bnb_func


def single_head_attn_activation_patching():
    pass


def replace_activations_and_evaluate_logits_on_each_head(prompt, answer, avg_activations, model_wrapper,
                                                         last_token_only=True,
                                                         logic_len=None):
    """
    Replace the activations at each head of each layer and evaluate the logits improvement.
    :param prompt:
    :param answer:
    :param avg_activations:
    :param model_wrapper:
    :param last_token_only:
    :param logic_len:
    :return: indirect effect of each layer and head: tensor Shape: (n_layers, n_heads)
    """
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    model_config = model_wrapper.model_config
    model = model_wrapper.model

    # Tokenize the prompt and get the word indices
    inputs = tokenizer(prompt, is_split_into_words=True, return_tensors='pt').to(device)
    word_idx = inputs.word_ids(batch_index=0)
    # Get real sequence length
    seq_len = len(word_idx)

    # Align word indices with the logic slots
    if word_idx[0] is None:
        word_idx = [0 if i == 0 else idx + 1 for i, idx in enumerate(word_idx)]

    # Allocate space for the indirect effect storage
    n_layers = model_config['n_layers']
    n_heads = model_config['n_heads']
    n_classes = 1 if last_token_only else logic_len
    indirect_effect_storage = torch.zeros(n_layers, n_heads).to(device)  # Shape: (n_layers, n_heads)

    # Calculate clean prompt probability baseline
    with torch.no_grad():
        clean_logits = model(**inputs).logits[:, -1, :]  # Shape: (1, vocab_size)
        clean_probs = torch.softmax(clean_logits, dim=-1)  # Shape: (1, vocab_size)

    # Replace the activations at each head of each layer
    for layer in range(n_layers):
        head_hook_layer = [model_config['attn_hook_names'][layer]]

        for head_n in range(model_config['n_heads']):
            tokens_idx = [-1] if last_token_only else list(range(seq_len))
            intervention_locations = [(layer, head_n, idx) for idx in tokens_idx]

            # Create a function to replace the activations
            intervention_fn = replace_activation_on_single_head(intervention_locations, avg_activations, word_idx,
                                                                model)

            # Edit the output of the proj layer
            with TraceDict(model, layers=head_hook_layer, edit_output=intervention_fn) as td:
                output = model(**inputs).logits[:, -1, :]  # Shape: (1, 1, vocab_size)

            # Get the first token_id of the answer tokens
            answer_token_id = get_answer_token_id(answer, tokenizer).to(device)  # Type: torch.LongTensor

            # Evaluate improvement of probs of answer tokens
            # Convert to probability distribution
            intervention_probs = torch.softmax(output, dim=-1)  # Shape: (1, vocab_size)
            indirect_effect_storage[layer, head_n] = (intervention_probs - clean_probs).index_select(1,
                                                                                                     answer_token_id).squeeze()

    return indirect_effect_storage


def replace_activation_on_single_head(intervention_locations, avg_activations, word_idx, model_wrapper):
    """
    Create a function to replace the activations at a single head of each layer.
    :param intervention_locations: replacement locations
    :param avg_activations:
    :param word_idx:
    :param model_wrapper:
    :return: intervention function
    """
    edit_layers = [x[0] for x in intervention_locations]
    model_config = model_wrapper.model_config
    model = model_wrapper.model

    def rep_act(output, layer_name, inputs):
        # Extract current layer number
        current_layer = int(layer_name.split('.')[2])

        # Replace activations only in the edit layers
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            original_shape = inputs.shape  # Shape: (batch_size, seq_len, hidden_size)

            # Split hidden_size into (n_heads, head_dim)
            new_shape = inputs.size()[:-1] + (
                model_config['n_heads'],
                model_config['hidden_size'] // model_config['n_heads']
            )  # Shape: (batch_size, seq_len, n_heads, head_dim)

            inputs = inputs.view(*new_shape)

            # ======== Begin Replace ========
            # Replace the activations at the specified locations in the intervention locations list
            for (layer, head_n, token_idx) in intervention_locations:
                if layer == current_layer:
                    inputs[-1, token_idx, head_n] = avg_activations[layer, head_n, word_idx[token_idx]]
            # ======== Finish Replace ========

            # Reshape back to original shape
            inputs = inputs.view(*original_shape)  # Shape: (batch_size, seq_len, hidden_size)

            # Get the current hooked attention projection module
            proj_module = get_module(model, layer_name)
            out_proj = proj_module.weight  # W_O matrix

            if 'gemma' in model_config['name_or_path']:
                new_output = torch.matmul(inputs, out_proj.T)
            elif 'llama' in model_config['name_or_path']:
                # if LLaMA quantized to 4bit, then dequantize first
                if model.is_loaded_in_4bit:
                    out_proj_dequant = bnb_func.dequantize_4bit(
                        out_proj.data,
                        out_proj.quant_state
                    )
                    new_output = torch.matmul(inputs, out_proj_dequant.T)
                else:
                    new_output = torch.matmul(inputs, out_proj.T)  # Shape: (batch_size, seq_len, hidden_size)

            # Return the new output
            return new_output
        else:
            # Directly return the original output if not in the edit layers
            return output

    return rep_act


def get_answer_token_id(answer, tokenizer):
    """
    Get the first valid token ID of the answer.
    :param answer: str
    :param tokenizer:
    :return: target token ID: torch.LongTensor
    """
    # Tokenize the answer and get the token ID
    answer_tokens = tokenizer(answer, add_special_tokens=False, return_tensors='pt').input_ids[0]

    # Get the token ID of the answer
    target_token_id = answer_tokens[0].item()  # Get the first valid token ID

    return torch.LongTensor([target_token_id])
