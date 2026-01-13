import json
import time
import yaml
from copy import deepcopy
from typing import Optional, Type, TypeVar, List
from pydantic import BaseModel
from openai import OpenAI

# Toggle between OpenAI and Ollama
llm_openai_api = True  # or False, depending on usage
T = TypeVar('T', bound=BaseModel)


class ContextHandlerInput(BaseModel):
    old_context: str
    new_context: str
    storage_short_content: List[str]
    storage_short_time_difference: int


class ContextHandlerOutput(BaseModel):
    adapted_context: str
    replanning_decision: bool


class ContextStore:
    def __init__(self):
        self.context = {}
        self.id_count = 0
        self.storage_short = {"plan": [], "time": int(time.time())}
        self.old_context = ""

        self.prompt_data = self.load_config("prompts_v2.yaml")
        self.robot_data = self.load_config("robot_pepper/robot_data.yaml")

        if llm_openai_api:
            self.client = OpenAI()
        else:
            self.llm_name = "llama3.2"

    def update_context(self, new_context):
        print("[context]=== update context ===")
        self.context = new_context.copy()
        self.context["id"] = deepcopy(self.id_count)
        print(f"[context].   Context updated: {self.context}")
        self.id_count += 1

    def update_storage_short(self, plan):
        self.storage_short["time"] = time.time()
        self.storage_short["plan"] = plan

    def handle_context_input(self, input_context):
        context = input_context.copy()

        print("[context]=== handle context ===")
        print("[context].   old_context: ", str(self.old_context))
        print("[context].   new_context: ", str(context))
        print("[context].   storage_short_content: ", self.storage_short["plan"])
        print("[context].   storage_short_time_difference: ", (int(time.time()) - int(self.storage_short["time"])))

        if context["type"] == "text":
            input_data = ContextHandlerInput(
                old_context=self.old_context,
                new_context=str(context["context"]),
                storage_short_content=self.storage_short["plan"],
                storage_short_time_difference=abs(int(time.time()) - int(self.storage_short["time"]))
            )
            system_prompt = self.prompt_data["context_handler_text"]

            result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextHandlerOutput)
            self.old_context = str(context["context"])

            print("[context].   result: ", result)
            if result.replanning_decision:
                self.update_context({"context": result.adapted_context})

        elif context["type"] in {"image", "image_path"}:
            print("Image handling not yet implemented in chat API version.")
        else:
            print("WARNING: Cannot handle context of type ", context["type"])

    def get_context(self):
        return self.context.copy()

    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

    def call_llm_with_schema(
            self,
            system_prompt: str,
            input_data: dict,
            output_schema: Type[T],
            temperature: float = 0.0,
            image_path: Optional[str] = None,
        ) -> T:
        """
        Calls LLM with optional image input and structured schema output using Chat Completions API.
        """
        if llm_openai_api:
            user_prompt = json.dumps(input_data)

            if image_path:
                file_id = self.upload_image_for_vision(image_path)
                user_content = [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"openai://file/{file_id}"
                        }
                    }
                ]
            else:
                user_content = user_prompt

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            print(f"[context].  --- Calling OpenAI Chat API{' with image' if image_path else ''}")
            response = self.client.beta.chat.completions.parse(
                model="gpt-4.1",
                messages=messages,
                response_format=output_schema,
                temperature=temperature,
                store=True,
            )

            raw_output = response.choices[0].message.content.strip()
            try:
                return output_schema.model_validate_json(raw_output)
            except Exception as e:
                raise ValueError(f"Failed to parse model output into schema: {e}\n\nOutput:\n{raw_output}")

        else:
            # Fallback to Ollama
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(input_data)}
            ]
            response = ollama.chat(
                model=self.llm_name,
                messages=messages,
                format=output_schema.model_json_schema(),
                options={'temperature': temperature}
            )
            return output_schema.model_validate_json(response.message.content)

    def upload_image_for_vision(self, image_path: str) -> str:
        # Placeholder: You’ll need to implement actual OpenAI file upload
        raise NotImplementedError("Image uploading not implemented yet.")


ctx = ContextStore()
ctx.handle_context_input({
    "type": "text",
    "context": "The robot just picked up the cup from the table."
})
print("Current context:", ctx.get_context())

