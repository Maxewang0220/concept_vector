import torch
from baukit import TraceDict


def get_mean_head_activation(model_wrapper, dataset):
    """
    Get the whole mean head activations for a given model and dataset.
    :param model_wrapper: the model wrapper
    :param tokenizer: the corresponding tokenizer
    :param dataset: untokenized dataset
    :return: mean_activations Shape: (layers, heads, tokens, head_dim)
    """

    # get model config
    model_config = model_wrapper.model_config

    def split_activations_by_head(activations, model_config):
        # input shape: (batch_size, seq_len, n_heads * head_dim)
        new_shape = activations.size()[:-1] + (
            model_config['n_heads'],
            model_config['hidden_size'] // model_config['n_heads'])

        # output shape: (batch_size, seq_len, n_heads, hidden_size // n_heads)
        activations = activations.view(*new_shape)
        return activations

    prepend_bos_flag = model_config['prepend_bos']
    # get logic labels sequence length
    if model_config['prepend_bos']:
        logic_len = len(dataset[0]['prompt']) + 1
    else:
        logic_len = len(dataset[0]['prompt'])

    # allocate activation storage
    activation_storage = torch.zeros(
        len(dataset),
        model_config['n_layers'],
        model_config['n_heads'],
        logic_len,
        model_config['hidden_size'] // model_config['n_heads']
    )  # Shape: (batch_size, n_layers, n_heads, logic_seq_len, head_dim)

    for n in range(len(dataset)):
        activations_td, word_idx = extract_attn_activations(
            tokens=dataset[n]['prompt'],
            layers=model_config['attn_hook_names'],
            model_wrapper=model_wrapper
        )  # Shape: (batch_size=1, seq_len, n_heads * head_dim)

        # Map the tokens' activations to the corresponding word indices
        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_config) for layer in
                                      model_config['attn_hook_names']]).permute(0, 2, 1,
                                                                                3)  # Shape: (n_layers, n_heads, seq_len, head_dim)

        # Create a new tensor with logic slots to store the activations
        n_layers, n_heads, seq_len, head_dim = stack_initial.shape
        stack_filtered = torch.zeros((n_layers, n_heads, logic_len, head_dim), device=stack_initial.device,
                                     dtype=stack_initial.dtype)

        i = 0

        for j in range(logic_len):
            # <bos> slot 0
            if prepend_bos_flag and j == 0:
                stack_filtered[:, :, j, :] = stack_initial[:, :, 0, :].view(n_layers, n_heads, head_dim)
                i += 1
                continue

            # record the start and end index of the segment
            start = i
            while i < seq_len and (
                    (word_idx[i] + 1 == j) if prepend_bos_flag
                    else word_idx[i] == j):
                i += 1
            end = i

            if start == end:
                segment = stack_initial[:, :, start, :]
            else:
                # take the mean of the activations for the segment
                segment = stack_initial[:, :, start:end, :]  # Shape: (n_layers, n_heads, end-start, head_dim)
            stack_filtered[:, :, j, :] = segment.mean(dim=2)  # Shape: (n_layers, n_heads, head_dim)

        # the n-th example activations
        activation_storage[n] = stack_filtered

    mean_activations = activation_storage.mean(dim=0)  # Shape: (n_layers, n_heads, logic_len, head_dim)
    return mean_activations


def extract_attn_activations(tokens, layers, model_wrapper):
    """
    Extract the attention activations for a given model and dataset.
    :param tokens:
    :param layers:
    :param model_wrapper:
    :return: the input activations of the proj layers
    """
    tokenizer = model_wrapper.tokenizer
    device = model_wrapper.device
    model = model_wrapper.model

    inputs = tokenizer(tokens, is_split_into_words=True, return_tensors='pt').to(device)
    word_idx = inputs.word_ids(batch_index=0)

    # Access Activations
    # Keep inputs of proj layers
    with TraceDict(model, layers=layers, retain_input=True, retain_output=False) as td:
        model(**inputs)  # Shape: (batch_size, seq_len, vocab_size)

    return td, word_idx
