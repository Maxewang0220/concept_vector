import datasets
import pandas
import pandas as pd
import os
import random
import json

# input-output pairs ICL example template
# example: Input: 2 -> Output: prime\n
ICL_EXAMPLE_TEMPLATE = """{input_prefix}{input}{separator_first}{output_prefix}{output}{separator_second}"""

# input-output pairs prompt template
# example: Input: number -> Output: parity\nInput: 3 -> Output: odd\nInput: 4 -> Output: \nDirectly output the answer.
PROMPT_TEMPLATE = """{ICL_examples}{query}{instruction}"""


class DatasetConfig:
    """
    Configuration class for dataset construction.
    Attributes:
    - n_shot (int): The number of examples to include in the prompt.
    - data_size (int): The size of the dataset.
    - is_save (bool): Whether to save the dataset to disk.
    - concept_input (str): The concept for the input.
    - concept_output (str): The concept for the output.
    - instruction (str): The instruction string.
    - input_prefix (str): The prefix for the input string.
    - output_prefix (str): The prefix for the output string.
    - separator_first (str): The separator between input and output.
    - separator_second (str): The separator after the output.
    """

    def __init__(self, n_shot, data_size, is_save, concept_input, concept_output, instruction, input_prefix,
                 output_prefix,
                 separator_first,
                 separator_second):
        self.n_shot = n_shot
        self.data_size = data_size
        self.is_save = is_save
        self.concept_input = concept_input
        self.concept_output = concept_output
        self.instruction = instruction
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.separator_first = separator_first
        self.separator_second = separator_second

    def __str__(self):
        return f"n_shot:{self.n_shot}, data_size:{self.data_size}, is_save:{self.is_save}, concept_input: {self.concept_input}, concept_output: {self.concept_output}, instruction: {self.instruction}, input_prefix: {self.input_prefix}, output_prefix: {self.output_prefix}, separator_first: {self.separator_first}, separator_second: {self.separator_second}"


def construct_ICL_example_str(input_prefix: str, input: str, output_prefix: str, output: str, separator_first: str,
                              separator_second: str) -> str:
    """
    Construct an basic ICL example from input and output strings.

    Parameters:
    - input_prefix (str): The prefix for the input string.
    - input (str): The input string.
    - output_prefix (str): The prefix for the output string.
    - output (str): The output string.
    - separator_first (str): The separator between input and output.
    - separator_second (str): The separator after the output.

    Returns:
    - str: The constructed ICL example.
    """
    return ICL_EXAMPLE_TEMPLATE.format(input_prefix=input_prefix, input=input, separator_first=separator_first,
                                       output_prefix=output_prefix, output=output, separator_second=separator_second)


def construct_prompt_str(ICL_examples: str, query: str, instruction: str) -> str:
    """
    Construct a prompt from ICL examples and an instruction.

    Parameters:
    - ICL_examples (str): The ICL examples string.
    - query (str): The query string. Question and empty answer.
    - instruction (str): The instruction string. Limit the output content to one word or other requirements.

    Returns:
    - str: The constructed prompt.
    """
    return PROMPT_TEMPLATE.format(ICL_examples=ICL_examples, query=query,
                                  instruction=instruction)


def construct_ICL_example(input_prefix: str, input: str, output_prefix: str, output: str, separator_first: str,
                          separator_second: str) -> list:
    """
    Construct an basic ICL example from input and output strings.

    Parameters:
    - input_prefix (str): The prefix for the input string.
    - input (str): The input string.
    - output_prefix (str): The prefix for the output string.
    - output (str): The output string.
    - separator_first (str): The separator between input and output.
    - separator_second (str): The separator after the output.

    Returns:
    - list: The split to logic words ICL example list.
    """
    ICL_examples_list = []
    if input_prefix:
        ICL_examples_list.append(input_prefix)
    if input:
        ICL_examples_list.append(input)
    if separator_first:
        ICL_examples_list.append(separator_first)
    if output_prefix:
        ICL_examples_list.append(output_prefix)
    if output:
        ICL_examples_list.append(output)
    if separator_second:
        ICL_examples_list.append(separator_second)

    return ICL_examples_list


def construct_prompt(ICL_examples: list, query: list, instruction: str) -> list:
    """
    Construct a prompt from ICL examples and an instruction.

    Parameters:
    - ICL_examples (list): The ICL examples list.
    - query (list): The query list. Question and empty answer.
    - instruction (str): The instruction string. Limit the output content to one word or other requirements.

    Returns:
    - list: The split to logic words prompt list.
    """
    prompt_list = []
    prompt_list.extend(ICL_examples)
    prompt_list.extend(query)
    prompt_list.append(instruction)

    return prompt_list


def construct_n_shot_dataset(dataset: pandas.DataFrame, dataset_config: DatasetConfig) -> datasets.Dataset:
    """
    Construct a dataset for n-shot learning.

    Parameters:
    - dataset (pandas.DataFrame): The input dataset.
    - n_shot (int): The number of examples to include in the prompt.
    - data_size (int): The size of the dataset.
    - dataset_config (PromptConfig): The configuration for the prompt.

    Returns:
    - datasets.Dataset: The constructed dataset.
    """
    prompt_list = []
    answer_list = []

    n_shot = dataset_config.n_shot
    data_size = dataset_config.data_size

    # Get all possible indices from the dataset
    all_indices = list(range(len(dataset)))

    for _ in range(data_size):
        # Randomly sample n_shot + 1 unique indices
        try:
            selected_indices = random.sample(all_indices, k=n_shot + 1)
        except ValueError:
            # Avoid n_shot + 1 larger than the dataset size
            selected_indices = random.choices(all_indices, k=n_shot + 1)

        example_indices = selected_indices[:-1]
        query_index = selected_indices[-1]

        # Create ICL examples
        ICL_examples = []

        # If concept is not None, add the concept example
        if dataset_config.concept_input:
            ICL_examples.extend(
                construct_ICL_example(input_prefix=dataset_config.input_prefix, input=dataset_config.concept_input,
                                      separator_first=dataset_config.separator_first,
                                      output_prefix=dataset_config.output_prefix, output=dataset_config.concept_output,
                                      separator_second=dataset_config.separator_second))

        # ICL examples
        for idx in example_indices:
            input = str(dataset.iloc[idx]['input'])
            output = str(dataset.iloc[idx]['output'])
            ICL_example = construct_ICL_example(input_prefix=dataset_config.input_prefix, input=input,
                                                separator_first=dataset_config.separator_first,
                                                output_prefix=dataset_config.output_prefix, output=output,
                                                separator_second=dataset_config.separator_second)
            ICL_examples.append(ICL_example)

        # Create the query
        query_input = str(dataset.iloc[query_index]['input'])
        query_output = str(dataset.iloc[query_index]['output'])
        query = construct_ICL_example(input_prefix=dataset_config.input_prefix, input=query_input,
                                      separator_first=dataset_config.separator_first,
                                      output_prefix=dataset_config.output_prefix, output="",
                                      separator_second=dataset_config.separator_second)

        # Construct the prompt
        instruction = dataset_config.instruction
        prompt = construct_prompt(ICL_examples=ICL_examples,
                                  query=query, instruction=instruction)
        answer = query_output

        # Append the prompt and answer to the lists
        prompt_list.append(prompt)
        answer_list.append(answer)

    # Create a dataset from the prompt and answer pairs
    dataset_dict = {
        'prompt': prompt_list,
        'answer': answer_list
    }
    dataset = datasets.Dataset.from_dict(dataset_dict)

    return dataset


def load_raw_dataset(dataset_name: str, dataset_config: DatasetConfig) -> datasets.Dataset:
    """
    Load a raw dataset from ../datasets directory

    Parameters:
    - name (str): The name of the dataset to load.
    - dataset_config (DatasetConfig): The configuration for the dataset.

    Returns:
    - datasets.DatasetDict: The loaded dataset.
    """
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, f"../datasets/{dataset_name}")

    n_shot = dataset_config.n_shot
    data_size = dataset_config.data_size
    is_save = dataset_config.is_save

    raw_dataset = pd.read_json(DATA_DIR)

    dataset = construct_n_shot_dataset(raw_dataset, dataset_config=dataset_config)

    if is_save:
        # Save the dataset to disk
        if dataset_config.concept_input:
            dataset_path = f"../datasets/processed/{dataset_name}_{n_shot}_{data_size}_c"
        else:
            dataset_path = f"../datasets/processed/{dataset_name}_{n_shot}_{data_size}"

        dataset.save_to_disk(os.path.join(BASE_DIR, dataset_path))
        print(
            f"Dataset saved to {os.path.join(BASE_DIR, dataset_path)}")

    for i in range(len(dataset)):
        print(f"{dataset[i]['prompt']}")
        print(f"{dataset[i]['answer']}")
        print()

    return dataset


def load_dataset(dataset_name: str) -> datasets.Dataset:
    """
    Load a processed dataset from ../datasets/processed directory

    Parameters:
    - name (str): The name of the dataset to load.

    Returns:
    - datasets.DatasetDict: The loaded dataset.
    """
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, f"../datasets/processed/{dataset_name}")

    dataset = datasets.load_from_disk(DATA_DIR)

    for i in range(len(dataset)):
        print(f"{dataset[i]['prompt']}")
        print(f"{dataset[i]['answer']}")
        print()

    return dataset


if __name__ == "__main__":
    # # test load raw dataset and generate processed dataset
    # dataset_name = "primality.json"
    #
    # dataset_config = DatasetConfig(
    #     n_shot=0,
    #     data_size=50,
    #     is_save=True,
    #     concept_input="present",
    #     concept_output="past",
    #     instruction="Directly output the answer.",
    #     input_prefix="Input: ",
    #     output_prefix="Output: ",
    #     separator_first=" -> ",
    #     separator_second="\n"
    # )
    #
    # random.seed(42)
    #
    # load_raw_dataset(dataset_name, dataset_config)

    # # test load processed dataset and metadata
    dataset_name = "primality.json_0_50_c"
    dataset = load_dataset(dataset_name)
    labels_len = len(dataset[0]['prompt'])
    print("dataset", dataset[0]['prompt'])
    print(f"labels_len: {labels_len}")
