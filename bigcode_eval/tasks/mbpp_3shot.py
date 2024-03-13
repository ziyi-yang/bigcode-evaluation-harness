"""3-shot evaluation for MBPP Sanitinized (Hand-Verified as in Mistral Tech Report)
Program Synthesis with Large Language Models
https://arxiv.org/abs/2108.07732

The benchmark consists of around 1,000 crowd-sourced Python programming problems, 
designed to be solvable by entry level programmers, covering programming fundamentals, 
standard library functionality, and so on. Each problem consists of a task description, 
code solution and 3 automated test cases. As described in the paper, a subset of the data
has been hand-verified by the authors.

Homepage:: https://github.com/google-research/google-research/tree/master/mbpp
"""
import json

from bigcode_eval.base import Task
from bigcode_eval.tasks.custom_metrics.code_eval import compute_code_eval

_CITATION = """
@article{austin2021program,
  title={Program Synthesis with Large Language Models},
  author={Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others},
  journal={arXiv preprint arXiv:2108.07732},
  year={2021}
}
"""
# Number of few shot examples to consider
NUM_SHOTS=3

class MBPP_3shot(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    DATASET_NAME = "sanitized"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 257
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        with open(
            "bigcode_eval/tasks/few_shot_examples/mbpp_few_shot_prompts.json",
            "r",
        ) as file:
            examples = json.load(file)
        return examples

    @staticmethod
    def few_shot_prompt(entry, problem_input, few_examples):
        """Three-shot prompt format for MBPP"""
        prompt = "You are an expert Python programmer. "
        for question, solution in zip(
            few_examples["questions"][:NUM_SHOTS], few_examples["solutions"][:NUM_SHOTS]
        ):
            prompt += f'''Q: {question}\n\n# solution in Python:\n\n\n"""{question}"""\n{solution}\n\n\n\n\n\n'''
        prompt += f"""Q: {problem_input}\n\n# solution in Python:\n\n\n"""
        return entry + prompt

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        3-shot demonstration.
        """
        
        # description = doc["text"]
        entry = ""
        problem_input = doc["prompt"]
        few_shot_examples = self.fewshot_examples()
        prompt = self.few_shot_prompt(entry, problem_input, few_shot_examples)
        print(prompt)
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results
