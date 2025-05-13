def get_mean_head_activation(model, tokenizer, dataset):
    """
    Get the whole mean head activations for a given model and dataset.
    :param model: the model wrapper
    :param tokenizer: the corresponding tokenizer
    :param dataset: untokenized dataset
    :return: mean_activations shape (layers, heads, tokens, head_dim)
    """

    def split_activations_by_head(activations, model_config):
        # input shape: (batch_size, seq_len, n_heads * head_dim)
        new_shape = activations.size()[:-1] + (
            model_config['n_heads'],
            model_config['hidden_size'] // model_config['n_heads'])

        # output shape: (batch_size, seq_len, n_heads, hidden_size // n_heads)
        activations = activations.view(*new_shape)
        return activations

    pass
