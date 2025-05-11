from models.gemma2 import Gemma2Model
from models.llama3 import Llama3Model

def load_model_and_tokenizer(model_name:str, device='cuda'):
    """
    Loads a huggingface model and its corresponding tokenizer

    Parameters:
    model_name: huggingface name of the model to load (e.g. "google/gemma-2-2b", "meta-llama/Llama-3.1-70B")
    device: 'cuda' or 'cpu'

    Returns:
    a model wrapper contains model, tokenizer, and model config
    """

    if 'gemma' in model_name.lower():
        model = Gemma2Model(model_name=model_name, device=device)
    elif 'llama' in model_name.lower():
        model = Llama3Model(model_name=model_name, device=device)

    return model