from copy import deepcopy, copy
import os
import time

#import ollama
import yaml
import json
import logging

from typing import Type, TypeVar
from pydantic import BaseModel, conlist
from typing import List, Dict, Union, Literal, Optional

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # dotenv not installed, will use system environment variables

from openai import OpenAI
llm_openai_api = True

class ContextStore:
    def __init__(self):
        self.context = {}
        self.id_count = 0
        self.storage_short = {"plan": [], "time": int(time.time())}
        self.old_context = ""

        #self.prompt_data = self.load_config("prompts_v3.yaml")
        #self.robot_data = self.load_config("robot_pepper/robot_data.yaml")

        if llm_openai_api:
            # Check if API key is available
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logging.warning("OPENAI_API_KEY not found in environment variables!")
                logging.warning("Context store will not be able to use OpenAI API.")
            else:
                self.client = OpenAI()  # Will use OPENAI_API_KEY from environment
                logging.info("OpenAI API initialized in ContextStore")
        else:
            #self.llm_name = "gemma3:27b"
            #self.llm_name = "llama3.3"
            self.llm_name = "llama3.2"
            #self.llm_name = "mistral-small3.1"
            #self.llm_name = "qwen3:30b-a3b"
            #self.llm_name = "qwen3:32b"

    def update_context(self, new_context):
        logging.info("[context]=== update context ===")
        self.context = new_context.copy()
        self.context["id"] = deepcopy(self.id_count)
        logging.info(f"[context].   Context updated: {self.context}")
        self.id_count += 1

    def update_storage_short(self, plan):
        self.storage_short["time"] = time.time()
        self.storage_short["plan"] = plan

    def handle_context_input(self, input_context):
        self.update_context(input_context)
        """ context = input_context.copy()

        logging.info("[context]=== handle context ===")
        logging.info("[context].   old_context: ", str(self.old_context))
        logging.info("[context].   new_context: ", str(context))
        logging.info("[context].   storage_short_content: ", self.storage_short["plan"])
        logging.info("[context].   storage_short_time_difference: ", (int(time.time()) - int(self.storage_short["time"])))

        if context["type"] == "text":
            input_data = ContextHandlerInput(
                old_context = self.old_context,
                new_context=str(context["context"]),
                storage_short_content=self.storage_short["plan"],
                storage_short_time_difference= abs((int(time.time()) - int(self.storage_short["time"])))
            )
            system_prompt = self.prompt_data["context_handler_text"]
            
            result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextHandlerOutput)

            self.old_context = str(context["context"])

            logging.info("[context].   result: ", result)
            if result.replanning_decision:
                self.update_context({"context": result.adapted_context})
        elif context["type"] == "image_path":
            pass
        elif context["type"] == "image":
            pass
        else:
            logging.info("WARNING: Cannot handle context of type ", context["type"]) """



    def get_context(self):
        return self.context.copy()

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)


    def call_llm_with_schema(
            self,
            system_prompt: str,
            input_data: dict,
            output_schema: Type[TypeVar('T', bound=BaseModel)],
            temperature: float = 0.0,
            image_path: Optional[str] = None,
        ) -> TypeVar('T', bound=BaseModel):
        """
        Calls LLM with optional image input and structured schema output.

        Args:
            system_prompt: Instructions for the LLM.
            input_data: Dict input (serialized to JSON).
            output_schema: Pydantic model for validating the response.
            temperature: Sampling temperature.
            image_path: Optional local path to image file.

        Returns:
            Parsed output as instance of output_schema.
        """
        user_content = [{"type": "input_text", "text": json.dumps(input_data)}]

        if image_path:
            file_id = self.upload_image_for_vision(image_path)
            user_content.append({
                "type": "input_image",
                "file_id": file_id
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        if llm_openai_api:
            logging.info(f"[context].  --- Calling OpenAI API{' with image' if image_path else ''}")
            response = self.client.responses.parse(
                model="gpt-4.1-mini",
                input=messages,
                temperature=temperature,
                text_format=output_schema,
            )
            return response.output_parsed
        else:
            response = ollama.chat(
                model=self.llm_name,
                messages=messages,
                format=output_schema.model_json_schema(),
                options={'temperature': temperature}
            )
            return output_schema.model_validate_json(response.message.content)

class ContextHandlerInput(BaseModel):
    old_context: str
    new_context: str
    storage_short_content: List[str]
    storage_short_time_difference: int


class ContextHandlerOutput(BaseModel):
    adapted_context: str
    replanning_decision: bool