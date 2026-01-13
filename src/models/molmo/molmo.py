"""molmo.py.

File for providing the Molmo model implementation.
"""
import logging
import os

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from src.models.base import ModelBase
from src.models.config import Config


class MolmoModel(ModelBase):
    """Molmo model implementation."""

    def __init__(self, config: Config) -> None:
        """Initialization of the molmo model.

        Args:
            config (Config): Parsed config
        """
        # initialize the parent class
        super().__init__(config)

    def _load_specific_model(self) -> None:
        """Overridden function to populate self.model."""
        # Get model config and extract checkpoint path if present
        model_config = getattr(self.config, 'model', {})
        checkpoint_path = model_config.get('checkpoint', None)

        # Filter out checkpoint from kwargs passed to from_pretrained
        model_kwargs = {k: v for k, v in model_config.items() if k != 'checkpoint'}

        # Load base model from HuggingFace
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, **model_kwargs, trust_remote_code=True
        )

        # Load custom checkpoint if provided
        if checkpoint_path:
            logging.info(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint_file = os.path.join(checkpoint_path, "model.pt")

            if not os.path.exists(checkpoint_file):
                raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

            # Load checkpoint state dict
            checkpoint_state_dict = torch.load(checkpoint_file, map_location='cpu')

            # Add "model." prefix to all keys to match HuggingFace format
            prefixed_state_dict = {f"model.{k}": v for k, v in checkpoint_state_dict.items()}

            # Load state dict into model
            missing_keys, unexpected_keys = self.model.load_state_dict(prefixed_state_dict, strict=False)

            if missing_keys:
                logging.warning(f"Missing keys in checkpoint: {missing_keys}")
            if unexpected_keys:
                logging.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

            logging.info("Custom checkpoint loaded successfully!")

    def _init_processor(self) -> None:
        """Initializes the processor."""
        # Filter out checkpoint parameter if present
        model_config = getattr(self.config, 'model', {})
        processor_kwargs = {k: v for k, v in model_config.items() if k != 'checkpoint'}

        self.processor = AutoProcessor.from_pretrained(
            self.config.model_path, **processor_kwargs, trust_remote_code=True
        )

    def _generate_prompt(self, prompt: str, add_generation_prompt: bool = True, has_images: bool = False) -> str:
        """Generates the Molmo model prompt which will not use the chat template.

        [Note from Martin] I'd hack these parameters a bit for gradio, follow Base.

        Args:
            prompt (str): The prompt to return, set by the config.
            add_generation_prompt (bool): Whether to add a start token of a bot
                response.
            has_images (bool): Whether the model has images or not.

        Returns:
            str: The prompt to return, set by the config.
        """
        return prompt

    def _generate_processor_output(self, prompt: str, img_path: str) -> dict:
        """Generate the processor argument to be input into the processor.

        Args:
            prompt (str): The generated prompt string with the input text and
                the image labels.
            img_path (str): The specified image path.

        Returns:
            dict: The corresponding processor arguments per image and prompt.

        Raises:
            ValueError: If no prompt is provided when required.
        """
        if img_path is None:
            raise ValueError('Molmo cannot have text-only generation.')

        # prepare the data inputs according to
        # https://huggingface.co/allenai/Molmo-7B-D-0924
        data_inputs = self.processor.process(
            images=[Image.open(img_path)],
            text=prompt
        )

        # move inputs to the correct device and make a batch of size 1
        return {
            k: v.to(self.config.device).unsqueeze(0)
            for k, v in data_inputs.items()
        }

    def _forward(self, data: dict) -> None:
        """Given some input data, performs a single forward pass.

        This function itself can be overriden, while _hook_and_eval
        should be left in tact.

        Args:
            data (dict): The given data tensor.
        """
        generation_config = self.config.forward
        with torch.no_grad():
            _ = self.model.generate_from_batch(
                data,
                GenerationConfig(**generation_config),
                tokenizer=self.processor.tokenizer
            )
        logging.debug('Completed forward pass...')
