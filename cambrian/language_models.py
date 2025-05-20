"""
Uses an open source LLM to generate new eye configurations
for a Cambrian Agent.
"""
from __future__ import annotations

import json
import random
from string import Template
from typing import Dict, List, Optional, Union

import torch
from loguru import logger
from ollama import ChatResponse, chat
from transformers import AutoModelForCausalLM, AutoTokenizer

# Options google/gemma-2-2b-it,
MODEL_NAME = "google/gemma-2-2b-it"
# Options: dolphin3, exaone3.5:2.4b
OLLAMA_MODEL = "exaone3.5:2.4b"
NUM_RETRIES = 25


class SystemPrompt:
    """
    System prompt provided to LLM to generate a structured response.

    The system prompt is a template that includes the substitution variables.
    """

    GEMINI_PROMPT = Template(
        """
        You are an AI Scientist. The agent's last fitness score was ${fitness_score}.
        Current eye configuration: ${eye_config}
        Instructions:
        - Do not change the sign of the eye configuration values.
        - The fitness score should ideally be a positive number > 100.
        - Lat and lon range should be in the range of [-90, 90] and [-180, 180] respectively.

        <start_of_turn>user
        Recommend a new eye configuration to maximize fitness, using evolutionary principles.
        The agent is trying to detect items in a 2D enviornment.

        Respond ONLY in this xml format provided below:
        <reasoning> [Your reasoning here] </reasoning>
        <eye_config>
        {{"num_eyes: [int, int]", "fov": [float, float], "resolution": [int, int], "lat_range": [float, float], "lon_range": [float, float]}}
        </eye_config>
        DO NOT include any other text or explanation. Do not hallucinate. Include all thinking
        <end_of_turn>user
        <start_of_turn>model
    """
    )

    PHI_PROMPT = Template(
        """
        <|system|>
        You are an AI Scientist. The agent's last fitness score was ${fitness_score}.
        Current eye configuration: ${eye_config}
        <|end|>
        <|user|>
        Recommend a new eye configuration to maximize fitness, using evolutionary principles.

        Respond ONLY in this xml format provided below:
        <reasoning> [Your reasoning here] </reasoning>
        <eye_config>
        {{"num_eyes": [int, int], "fov": [float, float], "resolution": [int, int], \n
        "lat_range": [float, float], "lon_range": [float, float]}}
        </eye_config>

        DO NOT include any other text or explanation. Do not hallucinate. Include all thinking
        within the <reasoning> tag.
        <|end|>
        <|assistant|>
        """
    )

    LLAMA_PROMPT = Template(
        """
        You are an AI Scientist. The agent's last fitness score was ${fitness_score}.
        Current eye configuration: ${eye_config}

        Respond ONLY in this xml format provided below:
        <reasoning> [Your reasoning here] </reasoning>
        <eye_config>
        {{"num_eyes": [int, int], "fov": [float, float], "resolution": [int, int], "lat_range": [float, float], "lon_range": [float, float]}}
        </eye_config>
        DO NOT include any other text or explanation. Do not hallucinate.
        """
    )

    def substitute(self, fitness_score: float, eye_config: Dict[str, list]) -> str:
        """
        Substitutes the fitness score and eye configuration in the system prompt.

        Args:
            fitness_score (float) : Fitness score of the agent in the environment.
            eye_config (Dict[str, list]) : Current eye configuration of the agent.

        Returns:
            str : System prompt with updated substitutions.
        """

        # Substitute the fitness score and eye configuration
        system_prompt: str = self.LLAMA_PROMPT.safe_substitute(
            fitness_score=fitness_score,
            eye_config=json.dumps(eye_config),
        )
        logger.debug(f"System prompt: {system_prompt}")
        return system_prompt


class UserPrompt:
    """
    Set a default user prompt & option to override it.
    """

    def __init__(self):
        self._prompt: Optional[str] = None

        self.default_prompt: str = """
            Recommend a new eye configuration to maximize fitness, using evolutionary principles.
            The agent is operating in an environment to detect items in a 2D environment. The goal
            of the agent is to maximize the fitness score.

            Instructions:
            - Do not change the sign of the eye configuration values.
            - Do not change the number of items in the list of each eye configuration.
            - Add eyes only if you must.
            """

    def __call__(self, user_choice: bool = False) -> str:
        """
        Returns the user prompt, based on the user preference.

        Args:
            user_choice (bool) : User preference to override the default prompt. False by default
        Returns:
            str : User prompt.
        """

        prompt: str = (
            input("Please enter a custom prompt for the LLM")
            if user_choice
            else self.default_prompt
        )

        self.prompt = prompt
        return self.prompt

    @property
    def prompt(
        self,
    ) -> str:
        """
        Returns the user prompt/
        """
        if self._prompt is None:
            raise ValueError("User prompt is not set. Please set it before proceeding.")
        return self._prompt

    @prompt.setter
    def prompt(self, prompt: str) -> None:
        """
        Sets the user prompt
        """
        if prompt is None:
            raise ValueError("User prompt cannot be None.")

        self._prompt = prompt


class LanguageManager:
    """
    Uses the Gemini LLM model to generate a new eye configuration for the Cambrian agent.
    """

    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.counter: int = 0
        self.model_name: str = MODEL_NAME
        self.system_prompt: object = SystemPrompt()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None

        # Setup llm
        self.setup_llm()

    def __call__(self, fitness_score: float, eye_config: Dict[str, list]) -> str:
        """
        Generates a new eye configuration for the agent using the LLM.

        Args:
            fitness_score (float) : Score received by the agent in the environment.
            eye_config (Dict[str, list]) : Last Eye configuration generated by LLM.

        Returns:
            eye_config (Dict[str, list]) : New eye configuration generated by LLM.
        """

        # Get the system prompt
        system_prompt: str = SystemPrompt().substitute(
            fitness_score=fitness_score,
            eye_config=eye_config,
        )

        # Generate a new eye configuration
        new_eye_cfg: str = self.generate_response(system_prompt=system_prompt)

        return new_eye_cfg

    def setup_llm(
        self,
    ) -> None:
        """
        Sets up the LLM Model and tokenizer.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            skip_special_tokens=True,
            padding="max_length",
            truncation=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        logger.info(f"Successfully loaded the model: {self.model_name}")

    def generate_response(
        self, system_prompt: str
    ) -> Dict[str, List[Union[float, int]]]:
        """
        Generates a new eye configuration for the agent using the LLM.
        Args:
            system_prompt (str) : System prompt to guide the LLM.

        Returns:
            Dict[str, List] : New eye configuration for the agent, to use for the next run.
        """

        # Tokenize the input
        inputs = self.tokenizer(system_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate the response
        outputs = self.model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.5,
        )

        decoded_op = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1] :],  # Skip input tokens
        )
        logger.debug(f"Response: {decoded_op}")

        # Parse the response & retry if error occurs
        try:
            new_eye_cfg: str = self.parse_response(raw_response=decoded_op)

        except Exception as e:
            logger.error(f"Error parsing the response : {e}")
            self.counter += 1

            if self.counter < NUM_RETRIES:
                logger.info("Retrying to generate the response.")
                return self.generate_response(system_prompt=system_prompt)

            raise RuntimeError(f"LLM failed to generate a valid response with {str(e)}")
        logger.info(f"New eye configuration: {new_eye_cfg}")
        return new_eye_cfg

    def parse_response(self, raw_response: str) -> Dict[str, list]:
        """
        Parses the response from the LLM to extract the new eye config.
        The parsed new eye config will be same as the one provided in the
        response structure.

        Args:
            raw_response (str) : Response from the LLM.

        Returns:
            Dict[str, list]: New eye configuration for the agent.
        """

        def fix_json(bad_json: str) -> str:
            """
            Removes '{{ }}' from the string and replaces it with '{ }'.
            """
            if bad_json.startswith("{{"):
                bad_json = bad_json[1:]

            if bad_json.endswith("}}"):
                bad_json = bad_json[:-1]

            return bad_json

        # extract the reasoning from the response
        reasoning: str = (
            raw_response.split("<reasoning>")[-1].split("</reasoning>")[0].strip()
        )
        logger.info(f"Reasoning: {reasoning}")

        # Extract the eye config from the response
        eye_config: str = (
            raw_response.split("<eye_config>")[-1].split("</eye_config>")[0].strip()
        )
        logger.debug(f"Eye config: {eye_config}")

        # Remove any '' characters from the start and end of the string
        eye_config = eye_config.replace("'", "")
        eye_config = eye_config.replace(" ", "")
        eye_config = fix_json(eye_config)

        parsed_response = json.loads(eye_config)
        return parsed_response


class OllamaManager(LanguageManager):
    """
    Uses Ollama LLM models to generate eye configuration.
    """

    def __init__(self):
        super().__init__()

        self.model_name: str = OLLAMA_MODEL

    def setup_llm(
        self,
    ) -> None:
        """
        Sets up the OLLAMA Model and tokenizer.
        """
        pass

    def generate_response(
        self, system_prompt: str
    ) -> Dict[str, List[Union[float, int]]]:
        """
        Generates a new eye configuration for the agent using the LLM.
        Args:
            system_prompt (str) : System prompt to guide the LLM.

        Returns:
            Dict[str, List] : New eye configuration for the agent, to use for the next run.
        """

        # Post to the OLLAMA API
        input: str = UserPrompt()
        ollama_response: ChatResponse = chat(
            model=self.model_name,
            messages=[
                {"role": "System", "content": system_prompt},
                {"role": "user", "content": input()},
            ],
            stream=False,
        )

        response = ollama_response.message.content
        logger.debug(f"Response: {response}")

        # Parse the response & retry if error occurs
        try:
            new_eye_cfg: str = self.parse_response(raw_response=response)

        except Exception as e:
            logger.error(f"Error parsing the response : {e}")
            self.counter += 1

            if self.counter < NUM_RETRIES:
                logger.info("Retrying to generate the response.")
                return self.generate_response(system_prompt=system_prompt)

            raise RuntimeError(f"LLM failed to generate a valid response with {str(e)}")

        return new_eye_cfg


@logger.catch
def main():
    """
    Main function to test the LanguageManager.
    """

    random.seed(0)
    torch.random.manual_seed(0)

    # Initialize the LLM
    llm = OllamaManager()

    # Initial eye configuration
    init_eye_cfg: Dict[str, List[Union[float, int]]] = {
        "num_eyes": [1, 1],
        "fov": [45, 45],
        "resolution": [1, 1],
        "lat_range": [-5, 5],
        "lon_range": [-90, 90.0],
    }

    # Open a log file to store results
    with open("llm_test.log", "w") as log_file:
        log_file.write("Run,Fitness-Score,Eye-Configs\n")

        # Run 100 generations
        current_eye_cfg = init_eye_cfg
        for run in range(1, 101):
            logger.info(f"Run: {run}")

            # Generate a random fitness score
            fitness_score = random.sample([-1000.0, 1000.0], 1)[0]

            # Generate a new eye configuration
            new_eye_cfg = llm(fitness_score=fitness_score, eye_config=current_eye_cfg)

            # Log the results
            log_file.write(f"{run},{fitness_score},{json.dumps(new_eye_cfg)}\n")

            # Update the current eye configuration for the next run
            current_eye_cfg = new_eye_cfg

    logger.info("Completed 100 runs. Results logged to llm_test_results.log.")


if __name__ == "__main__":
    main()
