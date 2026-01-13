import ollama
import yaml

def load_config(self, config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def call_llm(self, model, system_prompt, user_prompt):
    print("calling llm")
    response = ollama.chat(model=model, messages=[
    {
        'role': 'system',
        'content': system_prompt,
    },
    {
        'role': 'user',
        'content': user_prompt,
    },
    ],
    options = {
    'temperature': 0
    })
    return response['message']['content']


""" from pydantic import BaseModel

from ollama import chat

class 


# Define the schema for the response
class FriendInfo(BaseModel):
  name: str
  age: int
  is_busy: bool


class FriendList(BaseModel):
  friends: list[FriendInfo]


# schema = {'type': 'object', 'properties': {'friends': {'type': 'array', 'items': {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}, 'is_available': {'type': 'boolean'}}, 'required': ['name', 'age', 'is_available']}}}, 'required': ['friends']}
response = chat(
  model='llama3.2',
  messages=[{'role': 'user', 'content': 'I have two friends. The first is Ollama 22 years old busy saving the world, and the second is Alonso 23 years old and wants to hang out. Return a list of friends in JSON format'}],
  format=FriendList.model_json_schema(),  # Use Pydantic to generate the schema or format=schema
  options={'temperature': 0},  # Make responses more deterministic
)

# Use Pydantic to validate the response
friends_response = FriendList.model_validate_json(response.message.content)
print(friends_response) """

from typing import Type, TypeVar
from pydantic import BaseModel
from ollama import chat
import json

# Generic type for Pydantic model output
T = TypeVar('T', bound=BaseModel)

def call_llm_with_schema(
    system_prompt: str,
    input_data: dict,
    output_schema: Type[T],
    model_name: str = 'llama3.2',
    temperature: float = 0.0,
) -> T:
    """
    Generic function to call Ollama LLM with:
    - a system prompt (instructions),
    - input data as JSON,
    - output format defined by a Pydantic schema class.

    Returns validated output_schema instance.

    Args:
        system_prompt: The system prompt instructions.
        input_data: Input data dict (will be serialized to JSON).
        output_schema: Pydantic model class for output validation.
        model_name: Model to use (default 'llama3.1').
        temperature: Sampling temperature (default 0).

    Returns:
        Instance of output_schema with parsed model output.
    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': json.dumps(input_data)},
    ]

    response = chat(
        model=model_name,
        messages=messages,
        format=output_schema.model_json_schema(),
        options={'temperature': temperature}
    )

    return output_schema.model_validate_json(response.message.content)

def load_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)


from pydantic import BaseModel, conlist
from typing import List, Dict, Union, Literal, Optional



class ReactionSteps(BaseModel):
    steps: List[str]

# output step 1
class HumanBehaviorInput(BaseModel):
    context: str

# output step 1
class HumanBehaviorResponse(BaseModel):
    behaviors: List[str]

# input step 2
class RobotTranslationInput(BaseModel):
    what_human_would_do: List[str]
    robot_capabilities: str

#example
""" # Create input data using the Pydantic model
input_data = RobotTranslationInput(
    what_human_would_do=[
        "Make eye contact to acknowledge the person's gaze."
    ],
    robot_capabilities=(
        "Body: The body can tilt forward and backward on knee and pelvis. "
        "The body can also tilt left and right on pelvis. "
        "Arms: The arms can be moved in all directions and can be bent. "
        "Head: The head can be rotated and tilted up and down. "
        "Speech: The robot cannot speak or play audio messages, but only move. "
        "Hearing: The robot cannot hear audio, but only move."
    )
)

# Convert to dict for use with the function
result = call_llm_with_schema(system_prompt, input_data.model_dump(), RobotProcedureResponse)

# Print output
print(result.robot_procedure)
 """

# output step 2
class RobotProcedureResponse(BaseModel):
    robot_procedure: List[str]

# Input step 3
class PrimitiveSequencingInput(BaseModel):
    robot_procedure: List[str]
    available_motion_primitives: Dict[str, str]

# Output step 3
class PrimitiveSequenceResponse(BaseModel):
    motion_sequence: List[str]

# input step 4
class MotionExpressivityInput(BaseModel):
    motion_sequence: List[str]
    context: str

# output step 4
class MotionExpressivityResponse(BaseModel):
    animation_descriptions: Dict[str, List[str]]


class PrincipleScale(BaseModel):
    type: str  # "int" or "categorical_list"
    range: Optional[List[int]] = None
    options: Optional[List[str]] = None
    length: Optional[int] = None

class PrincipleDefinition(BaseModel):
    description: str
    scale: PrincipleScale

# input step 5
class AnimationRatingInput(BaseModel):
    primitive: str
    animation_description: List[str]
    principles: Dict[str, PrincipleDefinition]

TimingPacingList = conlist(Literal["slow", "moderate", "fast"], min_length=3, max_length=3)

# output step 5
class AnimationRatingResponse(BaseModel):
    Anticipation: int
    Follow_Through_and_Overlapping_Action: int
    Arcs: int
    Timing_Pacing: TimingPacingList  # now strictly enforced
    Timing_Length: int
    Exaggeration: int




# full run
system_prompts = load_config("prompts_v2.yaml")
robot_data = load_config("robot_pepper/robot_data.yaml")

#step 1
print("step 1")
context_str = "Greet the person that enters the room through the door to your right"
input_data = HumanBehaviorInput(context=context_str)
system_prompt = system_prompts["context_to_human_actions"]
result = call_llm_with_schema(system_prompt, input_data.model_dump(), HumanBehaviorResponse)
print(result)

#step 2
print("step 2")
input_data = RobotTranslationInput(
    what_human_would_do=result.behaviors,
    robot_capabilities=robot_data["capabilities"]
)
system_prompt = system_prompts["human_actions_to_robot_actions"]
result = call_llm_with_schema(system_prompt, input_data.model_dump(), RobotProcedureResponse)
print(result)

#step 3
print("step 3")
input_data = PrimitiveSequencingInput(
    robot_procedure=result.robot_procedure,
    available_motion_primitives=robot_data["primitive_lib"]
)
system_prompt = system_prompts["robot_actions_to_sequence"]
result = call_llm_with_schema(system_prompt, input_data.model_dump(), PrimitiveSequenceResponse)
print(result)

#step 4
print("step 4")
input_data = MotionExpressivityInput(
    motion_sequence=result.motion_sequence,
    context=context_str
)
system_prompt = system_prompts["context_and_sequence_to_animation_description"]
result = call_llm_with_schema(system_prompt, input_data.model_dump(), MotionExpressivityResponse)
print(result)

#step 5
print("step 5")
for primitive_str in result.animation_descriptions:
    input_data = AnimationRatingInput(
        primitive=primitive_str,
        animation_description=result.animation_descriptions[primitive_str],
        principles=system_prompts["principles"]
    )
    system_prompt = system_prompts["animation_description_to_animation_principle_description"]
    final_result = call_llm_with_schema(system_prompt, input_data.model_dump(), AnimationRatingResponse).dict()
    print(final_result)






