import torch

from utils.intervention_utils import replace_activations_and_evaluate_logits_on_each_head


def calculate_casual_indirect_effect(dataset, mean_activations, model_wrapper, last_token_only=True):
    """
    Calculate the causal indirect effect of a model on the [clean ICL prompts/concept instruction] dataset.
    :param dataset: [corrupted ICL prompts/empty concept instruction] dataset
    :param mean_activations: the extracted activations from the clean prompts
    :param model_wrapper: subject model wrapper
    :param last_token_only: whether to only consider the last token of the prompt
    :return: the causal indirect effect of each layer and head, Shape: (batch_size, n_layers, n_heads)
    """
    model_config = model_wrapper.model_config
    model = model_wrapper.model
    n_trials = len(dataset)

    # get logic labels sequence length
    if model_config['prepend_bos']:
        logic_len = len(dataset[0]['prompt']) + 1
    else:
        logic_len = len(dataset[0]['prompt'])

    if last_token_only:
        indirect_effects = torch.zeros(n_trials,
                                       model_config['n_layers'],
                                       model_config['n_heads'])  # Shape: (batch_size, n_layers, n_heads)
    else:
        # replace all the logic slots activations
        indirect_effects = torch.zeros(n_trials,
                                       model_config['n_layers'],
                                       model_config['n_heads'],
                                       logic_len)  # Shape: (batch_size, n_layers, n_heads, logic_len)

    for i in range(n_trials):
        ind_effect = replace_activations_and_evaluate_logits_on_each_head(
            prompt=dataset[i]['prompt'],
            answer=dataset[i]['answer'],
            avg_activations=mean_activations,
            model_wrapper=model_wrapper,
            last_token_only=last_token_only,
            logic_len=logic_len)

        indirect_effects[i] = ind_effect.squeeze()  # Shape: (1, n_layers, n_heads) -> (n_layers, n_heads)

    return indirect_effects
