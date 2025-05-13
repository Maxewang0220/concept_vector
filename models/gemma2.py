import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.logging_utils import logger


class Gemma2Model:
    """
    Creating a Gemma2 model wrapper:
        "google/gemma-2-2b"
        "google/gemma-2-9b"
        "google/gemma-2-27b"
    """

    def __init__(self, model_name: str, device: str = 'cuda'):
        self.model_name = model_name
        self.device = device

        # huggingface config
        hf_kwargs = {}
        hf_kwargs["offload_folder"] = "./offload"
        # TODO: check if sdpa attention is available
        # hf_kwargs["attn_implementation"] = "eager"

        # Load the huggingface pretrained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **hf_kwargs
        ).to(device)

        # ensure pad_token setting
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model_config = {
            "n_heads": self.model.config.num_attention_heads,
            "n_layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "name_or_path": self.model.config._name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in
                                range(self.model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(self.model.config.num_hidden_layers)],
            "prepend_bos": True
        }

        self._log_memory_usage()

    def _log_memory_usage(self):
        for d in range(torch.cuda.device_count()):
            t = torch.cuda.get_device_properties(d).total_memory
            r = torch.cuda.memory_reserved(d)
            a = torch.cuda.memory_allocated(d)
            logger.info(
                f"Device {d}, total_memory: {t / 8 / 1024 / 1024:.4}Gb, reserved: {r / 8 / 1024 / 1024:.4}Gb, allocated: {a / 8 / 1024 / 1024:.4}Gb, free: {(t - r) / 8 / 1024 / 1024:.4}Gb")
