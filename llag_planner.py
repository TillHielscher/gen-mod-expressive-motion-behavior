#import ollama
import os
import yaml
import json
import re
import logging
from typing import Type, TypeVar
from pydantic import BaseModel, conlist
from typing import List, Dict, Union, Literal, Optional
from openai import OpenAI, APIError, APIConnectionError, RateLimitError

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file from current directory
except ImportError:
    pass  # dotenv not installed, will use system environment variables

from llag_block import LLAGTimelineBlock

llm_openai_api = True

class LLAGPlanner:

    def __init__(self, robot, prompt_data_path="prompts_v4.yaml"):
        """Initialize LLAGPlanner with a robot instance.
        
        Args:
            robot: Robot instance that implements RobotBase
            prompt_data_path: Path to prompt configuration YAML
        """
        logging.info("Planner init")
        self.robot = robot
        self.prompt_data = self.load_config(prompt_data_path)
        api_confirmation = input("Use OpenAI API? (y/n): ").strip().lower()
        if api_confirmation == 'y':
            # Check if API key is available
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                logging.error("OPENAI_API_KEY not found in environment variables!")
                exit(1)
            
            self.client = OpenAI()  # Will use OPENAI_API_KEY from environment
            #self.model_name = "gpt-4.1-mini"
            #self.model_name = "gpt-4.1"
            #self.model_name = "ft:gpt-4.1-nano-2025-04-14:tillkisir:20250710v1:BriK76uw"
            self.model_name = "ft:gpt-4.1-nano-2025-04-14:tillkisir:ft-v1:CDsmIPYb"
            logging.info(f"OpenAI API initialized successfully with model: {self.model_name}")
        else:
            logging.warning("Cancelled - OpenAI API not enabled")
            exit()

    def short_pipeline(self, context):
        mapped_animation_sequence, unmapped_animation_sequence = self.generate_animated_sequence(context)
        return mapped_animation_sequence, unmapped_animation_sequence

    def generate_animated_sequence(self, passed_context):
        system_prompt = self.prompt_data["plan_with_context"]
        logging.info("[planner]=== generate animated sequence ===")
        if "image" in passed_context or "text" in passed_context or "audio" in passed_context:
            logging.info("[planner]=== multimodal context detected ===")
            result =  self.multimodal_call(
                passed_system_prompt=system_prompt,
                context_dict=passed_context,
            )
        else:
            context = passed_context.get("context", 0)
            input_data = PlanWithContextInput(
                context=str(context),
                robot_capabilities=self.robot.get_capabilities(),
                available_motion_primitives=self.robot.get_primitive_lib(),
                principles=self.prompt_data["principles"]
            )
            
            result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), PlanWithContextOutput)
        mapped_animation_sequence = []
        unmapped_animation_sequence = []
        for animation in result.animated_sequence:
            mapped_animation = self.map_animational_principle_description_to_parameters(animation.dict())
            mapped_animation["motion_primitive"] = animation.motion_primitive
            mapped_animation_sequence.append(mapped_animation)
            unmapped_animation_sequence.append(animation.dict())

        return mapped_animation_sequence, unmapped_animation_sequence





    def long_pipeline(self, context):
        logging.info("[planner]=== long pipeline ===")
        logging.info("[planner]=== first call to get plan ===")
        timeline_block_list, plan = self.generate_plan_compact_structured(context)
        logging.info("[planner]=== second call to get animation descriptions ===")
        animation_descriptions = self.generate_animation_description_structured(plan, context)
        mapped_animation_sequence = []
        unmapped_animation_sequence = []
        for i, animation_description in enumerate(animation_descriptions):
            logging.info(f"[planner]=== third call to get animation parameters for {i+1}/{len(animation_descriptions)} ===")
            mapped_animation, unmapped_animation = self.generate_animation_parameters_structured(animation_description)
            #mapped_animation = self.map_animational_principle_description_to_parameters(call_result.dict())
            mapped_animation["motion_primitive"] = plan[i]
            mapped_animation_sequence.append(mapped_animation)
            unmapped_animation["motion_primitive"] = plan[i]
            unmapped_animation_sequence.append(unmapped_animation)
        return mapped_animation_sequence, unmapped_animation_sequence

    def generate_plan_compact_structured(self, context):
        context = context.get("context", 0)
        # Step 1
        input_data = ContextToSequenceInput(
            context=str(context),
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib()
        )
        system_prompt = self.prompt_data["context_to_sequence"]
        result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextToSequenceOutput)

        plan = result.motion_primitive_sequence

        timeline_block_list = []
        for primitive in plan:
            timeline_block_list.append(
                LLAGTimelineBlock(
                    primitive, 
                    primitive_path=self.robot.get_primitive_path(),
                    idle_data_yaml_path=self.robot.get_robot_description_path()
                )
            )

        return timeline_block_list, plan

    def generate_animation_description_structured(self, plan, context):
        context = context.get("context", 0)

        # Step 4
        input_data = ContextAndSequenceToAnimationDescriptionInput(
            motion_primitive_sequence=plan,
            context=context
        )
        system_prompt = self.prompt_data["context_and_sequence_to_animation_description"]
        result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextAndSequenceToAnimationDescriptionOutput)

        return result.animation_descriptions

    def generate_animation_parameters_structured(self, animation_description):
        logging.info("[planner]=== generate animation parameters ===")
        input_data = AnimationDescriptionToAnimationPrincipleDescriptionInput(
            motion_primitive=animation_description.motion,
            animation_description=animation_description.description,
            principles=self.prompt_data["principles"]
        )
        system_prompt = self.prompt_data["animation_description_to_animation_principle_description"]
        call_result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), AnimationDescriptionToAnimationPrincipleDescriptionOutput)
        mapped_animation = self.map_animational_principle_description_to_parameters(call_result.dict())
        unmapped_animation_sequence = call_result.dict()
        
        return mapped_animation, unmapped_animation_sequence





    

    def generate_animated_sequence_streaming(self, context):
        logging.info("[planner]=== generate animated sequence (streaming) ===")
        context = context.get("context", 0)

        input_data = PlanWithContextInput(
            context=str(context),
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib(),
            principles=self.prompt_data["principles"]
        )

        system_prompt = self.prompt_data["plan_with_context"]

        return self.call_llm_with_schema_streaming(
            system_prompt=system_prompt,
            input_data=input_data.model_dump(),
            output_schema=PlanWithContextOutput
        )


    def generate_plan_and_animation_description_compact_structured(self, context):
        logging.info("[planner]=== generate plan and animation description ===")
        context = context.get("context", 0)
        input_data = ContextToSequenceAndAnimationDescriptionInput(
            context=str(context),
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib()
        )
        system_prompt = self.prompt_data["context_to_sequence_and_animation_description"]
        if str(context) == "image":
            result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextToSequenceAndAnimationDescriptionOutput, image_path="scene_1.png")
        else:
            result = self.call_llm_with_schema(system_prompt, input_data.model_dump(), ContextToSequenceAndAnimationDescriptionOutput)

        plan = result.motion_primitive_sequence
        logging.info(f"[planner].  Plan: {plan}")

        timeline_block_list = []
        for primitive in plan:
            timeline_block_list.append(
                LLAGTimelineBlock(
                    primitive, 
                    primitive_path=self.robot.get_primitive_path(),
                    idle_data_yaml_path=self.robot.get_robot_description_path()
                )
            )

        #logging.info(plan)

        return timeline_block_list, plan, result.animation_descriptions

    



    def linear_map(self, value, old_min, old_max, new_min, new_max):
        if old_max == old_min:
            raise ValueError("Old range has zero length.")
        return new_min + (float(value - old_min) / (old_max - old_min)) * (new_max - new_min)

    def find_best_match(self, general_key, specific_keys):
        # Try exact match first
        if general_key in specific_keys:
            return general_key
        # Try partial match
        for spec_key in specific_keys:
            if spec_key in general_key:
                return spec_key
        raise KeyError(f"No match found for general key '{general_key}' in specific keys.")


    def map_animational_principle_description_to_parameters(self, animation_principle_description):
        mapped = {}

        # Include special specific fields directly
        if "Follow_Through_Data" in self.robot.get_parameter_ranges():
            mapped["Follow_Through_Data"] = self.robot.get_parameter_ranges()["Follow_Through_Data"]

        for key, value in animation_principle_description.items():
            if key not in self.prompt_data["principles"]:
                if key == "motion_primitive":
                    continue
                raise KeyError(f"{key} not found in general_ranges")

            # --- Special boolean case ---
            if key == "Slow_In_Slow_Out":
                if value == 0:
                    mapped[key] = False
                elif value == 1:
                    mapped[key] = True
                else:
                    raise ValueError(f"Invalid value for Slow_In_Slow_Out: {value}")
                continue

            # --- Require presence in robot_data before mapping ---
            if key not in self.robot.get_parameter_ranges():
                raise KeyError(f"{key} not found in specific parameter ranges")

            gen_min, gen_max = self.prompt_data["principles"][key]["scale_range"]
            spec_min = self.robot.get_parameter_ranges()[key]["min"]
            spec_max = self.robot.get_parameter_ranges()[key]["max"]

            # --- Principle-specific mapping rules ---
            if key in ["Timing", "Exaggeration", "Arcs"]:
                neutral = 1.0  # identity value (configurable if needed)
                if value == 0:
                    mapped[key] = neutral
                elif value < 0:
                    mapped[key] = neutral + (value / gen_min) * (spec_min - neutral)
                else:
                    mapped[key] = neutral + (value / gen_max) * (spec_max - neutral)

            elif key in ["Anticipation", "Follow_Through"]:
                # Standard linear mapping
                mapped[key] = spec_min + (value - gen_min) / (gen_max - gen_min) * (spec_max - spec_min)

            else:
                raise KeyError(f"No mapping defined for {key}")

            #print("Mapped", key, "value", value, "to", mapped[key])

        return mapped


    def map_animational_principle_description_to_parameters_only_linear_mapping(self, animation_principle_description):
        mapped = {}

        # Include special general fields directly
        if "Timing_Pacing" in self.prompt_data["principles"]:
            mapped["Timing_Pacing"] = self.prompt_data["principles"]["Timing_Pacing"]

        # Include special specific fields directly
        if "Follow_Through_Data" in self.robot.get_parameter_ranges():
            mapped["Follow_Through_Data"] = self.robot.get_parameter_ranges()["Follow_Through_Data"]

        for key, value in animation_principle_description.items():
            if key not in self.prompt_data["principles"]:
                if key == "motion_primitive":
                    continue
                raise KeyError(f"{key} not found in general_ranges")

            scale_info = self.prompt_data["principles"][key]["scale"]
            scale_type = scale_info["type"]

            if scale_type == "categorical_list":
                mapped[key] = value
            elif "range" in scale_info:
                gen_min, gen_max = scale_info["range"]
                if key not in self.robot.get_parameter_ranges():
                    raise KeyError(f"{key} not found in specific parameter ranges")
                spec_range = self.robot.get_parameter_ranges()[key]
                mapped_value = self.linear_map(value, gen_min, gen_max, spec_range["min"], spec_range["max"])
                mapped[key] = mapped_value
            else:
                raise ValueError(f"Unsupported or malformed scale type for '{key}': {scale_type}")

        return mapped

    def upload_image_for_vision(self, image_path: str) -> str:
        """
        Uploads a local image to OpenAI's Files API for vision input.

        Args:
            image_path: Path to image file on disk.

        Returns:
            file_id: The file ID returned by OpenAI.
        """
        with open(image_path, "rb") as file_content:
            result = self.client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    def upload_cv2_image(image_np) -> str:
        """
        Takes an OpenCV image (numpy array), encodes it, and uploads to OpenAI.

        Args:
            image_np: OpenCV image as a numpy array (e.g., from cv2.imread or cv2.VideoCapture).

        Returns:
            file_id: ID of the uploaded image for use in vision API.
        """
        # Encode image to JPEG in memory
        success, encoded_image = cv2.imencode('.jpg', image_np)
        if not success:
            raise ValueError("Failed to encode image")

        image_bytes = io.BytesIO(encoded_image.tobytes())
        image_bytes.name = "image.jpg"  # Required to mimic file-like object with name

        # Upload to OpenAI
        file = client.files.create(file=image_bytes, purpose="vision")
        return file.id

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
        logging.info(f"[planner].  --- Calling OpenAI API{' with image' if image_path else ''}")
        #print(f"[planner].  --- Calling OpenAI API{' with image' if image_path else ''}")

        #print the messages in a humnan readable way by indenting the json
        #print(json.dumps(messages, indent=2))


        #print("        Calling OpenAI API")
        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=messages,
                temperature=temperature,
                text_format=output_schema,
            )
        except APIError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass
        except Exception as e:
            print("Parsing failed:", e)
        #print("        Got Response")
        return response.output_parsed

    def multimodal_call(self, passed_system_prompt, context_dict):

        multimodal = False
        textual_context =""
        if "context" in context_dict and "audio" in context_dict or "context" in context_dict and "image" in context_dict or "audio" in context_dict and "image" in context_dict:
            textual_context = "The context consists of multiple modalities that complement each other. Indivudual modaties are separeted with a semicolon. "
            multimodal = True

        if "context" in context_dict:
            if multimodal:
                textual_context += "The text modality is as follows: "
            textual_context += context_dict["context"]
            if multimodal:
                textual_context += "; "
        if "audio" in context_dict:
            audio_file= open(context_dict.get("audio", None), "rb")
            transcription = self.client.audio.transcriptions.create(
                model="gpt-4o-transcribe", 
                file=audio_file,
                prompt="Transcribe the contents of the provided file. It can be speech, but it can also be sounds like a bang or knocking. All of which should be output in text."
            )
            if multimodal:
                textual_context += "The transcribed audio modality is as follows: "
            else:
                textual_context += "The context is given from transcribed speech as follows: "
            logging.warning(transcription)
            textual_context += transcription.text
            if multimodal:
                textual_context += "; "
        if "image" in context_dict:
            if multimodal:
                textual_context += "The image modality is given in the passed image for you to analyze."
            else:
                textual_context += "The context is given in the passed image for you to analyze."

        logging.warning(textual_context)
        print(textual_context)

        input_data = PlanWithContextInput(
            context=textual_context,
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib(),
            principles=self.prompt_data["principles"]
        )

        response = self.call_llm_with_schema(
            system_prompt=passed_system_prompt,
            input_data=input_data.model_dump(),
            output_schema=PlanWithContextOutput,
            temperature=0.0,
            image_path=context_dict.get("image", None)
        )

        return response

        

    def call_llm_with_schema_streaming(
        self,
        system_prompt: str,
        input_data: dict,
        output_schema: Type[TypeVar('T', bound=BaseModel)],
        temperature: float = 0.0,
        image_path: Optional[str] = None,
    ):
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

        logging.info(f"[planner]. --- Calling OpenAI API (streaming)")

        current_animations = []
        self.last_yielded_index = 0  # Track last yielded index for animations

        buffer = ""
        seen_hashes = set()

        item_regex = re.compile(
            r'\{[^{}]*"motion_primitive"\s*:\s*.*?"Exaggeration"\s*:\s*[^{}]+?\}'
        )

        with self.client.responses.stream(
            model=self.model_name,
            input=messages,
            temperature=temperature,
            text_format=output_schema,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    buffer += event.delta

                    for match in item_regex.finditer(buffer):
                        json_text = match.group(0)
                        try:
                            item = json.loads(json_text)
                            item_hash = hash(json_text)
                            if item_hash not in seen_hashes:
                                seen_hashes.add(item_hash)
                                yield item
                        except json.JSONDecodeError:
                            continue  # Incomplete or broken JSON — wait for next chunk

                elif event.type == "response.error":
                    raise RuntimeError(f"LLM streaming error: {event.error}")

            # Final parsed response
            final_response = stream.get_final_response()
            yield {"__final__": final_response.output_parsed}



    def load_config(self, config_file):
        with open(config_file, 'r') as file:
            return yaml.safe_load(file)

# Define Pydantic models for structured data
# general models

# input new step 1-3
class ContextToSequenceInput(BaseModel):
    context: str
    robot_capabilities: str
    available_motion_primitives: Dict[str, str]

# output new step 1-3
class ContextToSequenceOutput(BaseModel):
    motion_primitive_sequence: List[str]

# input step 4
class ContextAndSequenceToAnimationDescriptionInput(BaseModel):
    motion_primitive_sequence: List[str]
    context: str

class AnimationDescription(BaseModel):
    motion: str
    description: List[str]

# output step 4
class ContextAndSequenceToAnimationDescriptionOutput(BaseModel):
    animation_descriptions: List[AnimationDescription]

class AnimationDescription(BaseModel):
    motion: str
    description: List[str]

class PrincipleScale(BaseModel):
    type: str  # "int" or "categorical_list"
    range: Optional[List[int]] = None
    options: Optional[List[str]] = None
    length: Optional[int] = None

class PrincipleDefinition(BaseModel):
    description: str
    scale: PrincipleScale

class PrincipleScaleDefinition(BaseModel):
    scale_description: str
    scale_range: List[int]

TimingPacingList = conlist(Literal["slow", "moderate", "fast"], min_length=3, max_length=3)

class Animation(BaseModel):
    motion_primitive: str
    Anticipation: int
    Arcs: int
    Exaggeration: int
    Follow_Through: int
    Slow_In_Slow_Out: int
    Timing: int
    

class AnimationOld(BaseModel):
    motion_primitive: str
    Anticipation: int
    Follow_Through: int
    Arcs: int
    Timing_Pacing: TimingPacingList  # now strictly enforced
    Timing_Length: int
    Exaggeration: int


# specific models
# input step 1-4
class ContextToSequenceAndAnimationDescriptionInput(BaseModel):
    context: str
    robot_capabilities: str
    available_motion_primitives: Dict[str, str]

# output step 1-4
class ContextToSequenceAndAnimationDescriptionOutput(BaseModel):
    motion_primitive_sequence: List[str]
    animation_descriptions: List[AnimationDescription]

# input step 5
class AnimationDescriptionToAnimationPrincipleDescriptionInput(BaseModel):
    motion_primitive: str
    animation_description: List[str]
    principles: Dict[str, PrincipleScaleDefinition]

# output step 5
class AnimationDescriptionToAnimationPrincipleDescriptionOutput(BaseModel):
    Anticipation: int
    Arcs: int
    Exaggeration: int
    Follow_Through: int
    Slow_In_Slow_Out: int
    Timing: int

# old output step 5
class AnimationDescriptionToAnimationPrincipleDescriptionOutputOld(BaseModel):
    Anticipation: int
    Follow_Through: int
    Arcs: int
    Timing_Pacing: TimingPacingList  # now strictly enforced
    Timing_Length: int
    Exaggeration: int


# input step 1-5
class PlanWithContextInput(BaseModel):
    context: str
    robot_capabilities: str
    available_motion_primitives: Dict[str, str]
    principles: Dict[str, PrincipleScaleDefinition]

# output step 1-5
class PlanWithContextOutput(BaseModel):
    animated_sequence: List[Animation]