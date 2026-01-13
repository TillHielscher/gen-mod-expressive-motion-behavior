import ollama
from ollama import chat
from ollama import ChatResponse

import numpy as np

import yaml

        

def main(
    model: str = "llama3.2",
    #model: str = "mistral-small",
):
    
    
    gesture_lib_path = "../pepper-toolbox/gestures/pepper-core-anims-traj"

    promt_collection = load_prompt("prompt_config_reworked_v1.yaml")
    #system = config["system1"]
    #prompt = generate_prompt(config["prompt1"], description)

    #session = peppertoolbox.PepperSession()
    #session.motion_service.wakeUp()

    while True:
        promt_collection = load_prompt("prompt_config_reworked_v1.yaml")
        #scenario_input_str = input("Describe a scenario [type 'q' to quit execution]: ")
        scenario_input_str="The professor enters through the door at the back of the room."
        if scenario_input_str == "q":
                break
        
        #expressivity_input_str = input("Describe the expressivity the robots motion should carry [type 'q' to quit execution]: ")
        #expressivity_input_str="Comes in after the lecuture and is only looking to get something that was left behind."
        expressivity_input_str="Comes in urgent as he is slighly delayed for the first lecure you are looking forward to."
        #expressivity_input_str="You are tired and dragged down by the fact that the contents in pasts lectures were hard to follow."
        if expressivity_input_str == "q":
                break

        print("-------------- progress: ---------------")
        print("----------------- 0% -------------------")

        # scenario to human action
        system_prompt = promt_collection["action_scenario-to-human-action_system"]
        user_prompt = f"""[StartScenario]
        {scenario_input_str}
        [EndScenario]
        [StartContext]
        {expressivity_input_str}
        [EndContext]"""
        #human_action_str = call_llama(generator, temperature, top_p, system_prompt, user_prompt)
        human_action_str = call_llm(model, system_prompt, user_prompt)
        print(human_action_str)
        print("----------------- 20% ------------------")

        # human action to robot action
        system_prompt = promt_collection["action_human-to-robot-action_system"]
        user_prompt = human_action_str
        #robot_action_str = call_llama(generator, temperature, top_p, system_prompt, user_prompt)
        robot_action_str = call_llm(model, system_prompt, user_prompt)
        print(robot_action_str)
        print("----------------- 40% ------------------")

        # robot action to primitive sequence
        system_prompt = promt_collection["action_robot-action-to-primitive-sequence_system"]
        user_prompt = robot_action_str
        #primitive_sequence_str = call_llama(generator, temperature, top_p, system_prompt, user_prompt)
        primitive_sequence_str = call_llm(model, system_prompt, user_prompt)
        print(primitive_sequence_str)
        primitive_sequence_list = parse_primitivbe_sequence(primitive_sequence_str)
        print(len(primitive_sequence_list))
        #print(primitive_sequence_list)
        #print(primitive_sequence_str)
        print("----------------- 60% ------------------")

        # expressivity to animation description
        system_prompt = promt_collection["expressivity_description-to-animation-description_system"]
        user_prompt = f"""{primitive_sequence_str}
        [StartScenario]
        {scenario_input_str}
        [EndScenario]
        [StartExpressivity]
        {expressivity_input_str}
        [EndExpressivity]"""
        #animation_description_str = call_llama(generator, temperature, top_p, system_prompt, user_prompt)
        animation_description_str = call_llm(model, system_prompt, user_prompt)
        print(animation_description_str)
        individual_animation_description_list = parse_animation_description(animation_description_str)
        print(len(individual_animation_description_list))
        #print(individual_animation_description_list)
        print("----------------- 80% ------------------")        
        
        individual_parameter_list = []
        sequence_length = min(len(individual_animation_description_list), len(primitive_sequence_list))
        #print(sequence_length)
        #print(len(individual_animation_description_list))
        #print(individual_animation_description_list)
        #print(len(primitive_sequence_list))
        for i in range(sequence_length):
            individual_animation_description_str = individual_animation_description_list[i]
            primitive_str = primitive_sequence_list[i]
            # animation description to dmp parameters
            system_prompt = promt_collection["expressivity_animation-description-to-animation-principles_system"]
            user_prompt = f"""{individual_animation_description_str}
            [StartPrimitive]
            {primitive_str}
            [EndPrimitive]
            """
            #dmp_parameter_str = call_llama(generator, temperature, top_p, system_prompt, user_prompt)
            parameter_str = call_llm(model, system_prompt, user_prompt)
            print(parameter_str)
            print("--")
            individual_parameter_list.append(parameter_str)
        #print(individual_dmp_parameter_list)
        print("----------------- done -----------------")

        #print(primitive_sequence_str)
        #for i in range(len(individual_dmp_parameter_list)):
        #    print("Parameters for primitive: ")
        #    print(primitive_sequence_list[i])
        #    print(individual_dmp_parameter_list[i])

        print("========================================")

        break
        

        """ gesture_traj_demo = peppertoolbox.load_json_from_folder(gesture_lib_path, gesture_selection_str)

        gesture_traj_edmp_demo = peppertoolbox.translate_traj_pepper_edmp(gesture_traj_demo)

        while True:
            animation_description = input("Type a description for the motion, confirm with Enter [type 'q' to return to gesture selection]: ")

            if animation_description == "q":
                break

            edmp_params = call_llama(generator, animation_description, temperature, top_p)

            dmp = edmp.DMP(gesture_traj_edmp_demo, edmp_params["Arc"], 50, phase_type="spline", ap_timing_scaling=edmp_params["Timing"], ap_exaggeration=edmp_params["Exaggeration"], ap_anticipation=edmp_params["Anticipation"], ap_anticipation_n=2)
            dmp.canonical_system.calculate_spline(edmp_params["Slow In Slow Out"])

            gesture_traj_edmp, phase = dmp.run()

            #gesture_traj = peppertoolbox.translate_traj_edmp_pepper(gesture_traj_edmp, session.motion_service)

            #peppertoolbox.execute_traj(session, gesture_traj) """


def load_prompt(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

# Generate the final prompt
def generate_prompt(prompt_template, description):
    return prompt_template.replace("{description}", description)

def call_llm(model, system_prompt, user_prompt):
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
    #'temperature': 1.5, # very creative
    'temperature': 0 # very conservative (good for coding and correct syntax)
    })
    #print(response['message']['content'])
    return response['message']['content']


def call_llama(generator, temperature, top_p, system_promt, user_prompt):
    

    #description = animation_description
    

    dialogs = [
        [
        SystemMessage(content=system_promt),
        UserMessage(content=user_prompt),
        ],
    ]
    for dialog in dialogs:
        result = generator.chat_completion(
            dialog,
            #max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        #for msg in dialog:
        #    print(f"{msg.role.capitalize()}: {msg.content}\n")

        out_message = result.generation
        return out_message.content
        #print(out_message)
        #out_params_dict = eval("{" + out_message.content + "}")
        #print("-----------------------")
        #print("resulting parameter dict:")
        #print(out_params_dict)
        #print("\n==================================\n")
        #return out_params_dict


def parse_primitivbe_sequence(input_string):
    import re
    
    # Extract the part of the input between [Start sequence] and [End sequence]
    sequence_match = re.search(r"\[StartSequence\](.*?)\[EndSequence\]", input_string, re.S)
    if not sequence_match:
        return []
    
    # Get the sequence text and split it into individual lines
    sequence_text = sequence_match.group(1).strip()
    sequence_items = sequence_text.splitlines()
    
    # Return the sequence as a list of individual strings
    return sequence_items

def parse_animation_description(input_string):
    import re
    
    # Extract the part of the input that contains the animation descriptions
    animation_description = re.search(r"\[StartAnimationDescription\](.*?)\[EndAnimationDescription\]", input_string, re.S)
    if not animation_description:
        return []
    #print(animation_description.group(1).strip())
    animation_text = animation_description.group(1).strip()

    temp = animation_text.split("\n")
    parts = []
    for i in range(100):
        current_part = ""
        for snippet in temp:
            snippet = snippet.strip()
            if snippet.startswith(f"{i}."):
                current_part = current_part + snippet[2:-1] + "\n"
        if current_part != "":
            parts.append(current_part)
    
    return parts

def parse_animation_description2(input_string):
    import re
    
    # Extract the part of the input that contains the animation descriptions
    animation_description = re.search(r"\[StartAnimationDescription\](.*?)\[EndAnimationDescription\]", input_string, re.S)
    if not animation_description:
        return []
    
    animation_text = animation_description.group(1).strip()
    
    # Split the text into major parts by patterns like "1)", "2)", etc.
    major_parts = re.split(r"\n\d+\)", animation_text)
    
    # Remove empty strings and clean leading/trailing whitespace
    major_parts = [part.strip() for part in major_parts if part.strip()]
    
    # Re-attach the leading "x)" (1), 2), etc.) to each major part
    major_parts_with_steps = []
    major_index = 1
    for part in major_parts:
        # Find the sub-steps like "1.1)", "2.1)", etc., and clean their formatting
        sub_steps = re.findall(rf"{major_index}\.\d+\)\s*(.*)", part)
        if sub_steps:
            # Renumber sub-steps to be "1)", "2)", "3)", etc.
            reformatted_steps = [f"{i}) {step}" for i, step in enumerate(sub_steps, start=1)]
            # Add the steps as a single string
            major_parts_with_steps.append("\n".join(reformatted_steps))
        major_index += 1
    
    return major_parts_with_steps

def parse_animation_description1(input_string):
    import re
    
    # Extract the part of the input that contains the animation descriptions
    animation_description = re.search(r"\[StartAnimationDescription\](.*?)\[EndAnimationDescription\]", input_string, re.S)
    if not animation_description:
        return []
    
    animation_text = animation_description.group(1).strip()
    
    # Split the text into major parts by patterns like "1)", "2)", etc.
    lines = animation_text.split('\n')
    
    result = []
    current_list = None
    for line in lines:
        match = re.match(r"(\d+)\) for (.*)", line)
        if match:
            # New major part found, start a new list
            number, action = match.groups()
            current_list = []
            result.append((number + ') for ' + action, current_list))
        else:
            match = re.match(r"(\d+\.\d+)\) (.*)", line)
            if match:
                # Sub-step found, add to the current list
                number, description = match.groups()
                current_list.append(number + ') ' + description)
    
    return [descriptions for _, descriptions in result]
import re
def process_input(input_string):
    # Split input string into sections based on main headings (1), (2), etc.
    sections = re.split(r'\n(?=\d+\))', input_string)
    
    result = []
    for section in sections:
        lines = section.strip().split('\n')
        
        # Extract the subsections
        subsections = [line.strip() for line in lines[1:] if line.strip()]
        
        # If there are no subsections, skip this section
        if not subsections:
            continue
        
        # Format the subsections into a string with numbered items
        formatted_subsections = '\n'.join(f'{i+1}) {item}' for i, item in enumerate(subsections))
        
        result.append(formatted_subsections)
    
    return result



if __name__ == "__main__":
    main()
