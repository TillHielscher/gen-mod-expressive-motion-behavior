import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import logging
import json
from copy import deepcopy

from llag_planner import LLAGPlanner
from llag_timeline import LLAGTimeline
from llag_block import LLAGTimelineBlock
from llag_context import ContextStore
from llag_virtual_session import VirtualSession


def get_joint_idx_from_names(joint_name, joint_names_list):
    """Get the index of a joint name in a list of joint names."""
    try:
        return joint_names_list.index(joint_name)
    except ValueError:
        raise ValueError(f"Joint '{joint_name}' not found in joint names list")


class LLAGCore:
    def __init__(self, robot, use_real_robot=False, use_virtual_robot=True, console_log=True, debug_log=False, short_pipeline=True, modulate=True, prompt_data_path="prompts_v4.yaml"):
        """Initialize LLAGCore with a robot instance.
        
        Args:
            robot: Robot instance (e.g., PepperRobot, Go2Robot) that implements RobotBase
            use_real_robot: Whether to use real robot hardware
            use_virtual_robot: Whether to use virtual visualization
            console_log: Enable console logging
            debug_log: Enable debug logging
            short_pipeline: Use short pipeline mode
            modulate: Enable motion modulation
            prompt_data_path: Path to prompt configuration
        """
        self.robot = robot
        self.use_real_robot = use_real_robot
        self.use_virtual_robot = use_virtual_robot
        self.urdf_path = robot.get_urdf_path()
        self.console_log = console_log
        self.debug_log = debug_log
        self.short_pipeline = short_pipeline
        self.experiment_plan = []
        self.experiment_started = False
        self.experiment_finished = False
        self.rt_gain = 0.01
        self.rt_c = "0"  # real-time condition for controlled experiment: "0"=none, "2"=right, "3"=left-right
        self.modulate = modulate
        self.prompt_data_path = prompt_data_path
        
        # Get robot-specific paths from robot instance
        self.primitive_lib_path = robot.get_primitive_path()
        self.robot_description_path = robot.get_robot_description_path()

        # Logging setup
        if self.console_log:
            if self.debug_log:
                logging.basicConfig(level=logging.DEBUG)
            else:
                logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.WARNING)

        # Core components - pass robot to planner
        self.timeline = LLAGTimeline(self.primitive_lib_path, self.robot_description_path)
        self.planner = LLAGPlanner(robot, self.prompt_data_path)
        self.context_store = ContextStore()
        self.executor = ThreadPoolExecutor(max_workers=1)

        # Pepper/virtual sessions
        self.session = None
        self.virtual_session = None
        self.markdown_text = None
        self.viser_ready = False
        self.joint_names_list = None  # Will be populated from virtual session
        self.real_robot_exec_data = None  # Will store robot-specific execution data

    async def controller_loop(self, hz=60):
        interval = 1.0 / hz
        if self.use_real_robot:
            # Prepare real robot execution (robot-specific)
            self.real_robot_exec_data = self.robot.prepare_real_robot_execution(self.session)
        
        cycle_even = True
        while True:
            cycle_start = time.time()
            block = self.timeline.get_current_block()
            if block:
                block.step()
                block.update_goal(self.timeline.rt_data, self.rt_gain)
                block_state = block.dmp.get_state()

                if cycle_even:
                    # Execute on real robot if enabled
                    if self.use_real_robot:
                        self.robot.execute_state_on_real_robot(
                            self.session, 
                            block_state["y"],
                            self.real_robot_exec_data
                        )

                    # Execute on virtual robot if enabled
                    if self.use_virtual_robot:
                        self.robot.execute_state_on_virtual_robot(
                            self.virtual_session,
                            block_state["y"]
                        )

                if block.is_complete():
                    self.timeline.advance_block()

            elapsed = time.time() - cycle_start
            sleep_time = max(0, interval - elapsed)
            cycle_even != cycle_even
            await asyncio.sleep(sleep_time)

    async def context_listener(self):
        loop = asyncio.get_event_loop()
        while True:
            new_context = await loop.run_in_executor(None, input, "Enter new context (press ENTER to confirm): \n")
            if new_context.strip():
                context = new_context.strip()
                self.context_store.handle_context_input({"type": "text", "context": str(context)})
            await asyncio.sleep(0.5)

    async def planner_trigger_loop(self):
        last_context = None
        while True:
            if len(self.timeline.plan) == 0:
                if self.timeline.current_block.name_identifier.lower().startswith("zero") or self.timeline.current_block.name_identifier.lower().startswith("idle"):
                    if self.use_virtual_robot:
                        self.markdown_text.content = f"""Executing Idle.
                        
                        Plan has no items."""
                    else:
                        logging.debug(f"""Executing Idle.
                        
                        Plan has no items.""")
                    if self.experiment_started:
                        self.experiment_finished = True
                else:
                    if self.use_virtual_robot:
                        self.markdown_text.content = f"""Executing "{self.timeline.get_current_block().name_identifier}".
                        
                        Plan has no items."""
                    else:
                        logging.debug(f"""Executing "{self.timeline.get_current_block().name_identifier}".
                        
                        Plan has no items.""")
            else:
                if self.use_virtual_robot:
                    self.markdown_text.content = f"""Executing "{self.timeline.get_current_block().name_identifier}"
                
                    Remaining plan:\n{[block.name_identifier for block in self.timeline.plan]}"""
                else:
                    logging.debug(f"""Executing "{self.timeline.get_current_block().name_identifier}"
                
                    Remaining plan:\n{[block.name_identifier for block in self.timeline.plan]}""")
            context = self.context_store.get_context()
            if context != last_context and context != {}:
                last_context = context
                logging.info("[LLAGCore] Planner triggered with new context...")
                if self.use_virtual_robot:
                    self.markdown_text.content = f"Planning with new context"
                loop = asyncio.get_event_loop()
                if self.short_pipeline:
                    mapped_animation_sequence, unmapped_animation_sequence = await loop.run_in_executor(self.executor, self.planner.short_pipeline, context)
                else:
                    mapped_animation_sequence, unmapped_animation_sequence = await loop.run_in_executor(self.executor, self.planner.long_pipeline, context)

                logging.warning(f"[LLAGCore] Unmapped Animation Sequence: {json.dumps(unmapped_animation_sequence, indent=2)}")

                if len(self.timeline.plan) > 0:
                    logging.info("[LLAGCore] Clearing existing plan.")
                    self.timeline.plan = []

                
                
                for animation in mapped_animation_sequence:
                    block = LLAGTimelineBlock(animation["motion_primitive"], primitive_path=self.primitive_lib_path, idle_data_yaml_path=self.robot_description_path)
                    self.experiment_plan.append(deepcopy(block))
                    if self.modulate:
                        follow_through_data_list = []
                        for key in animation["Follow_Through_Data"].keys():
                            data = animation["Follow_Through_Data"][key]
                            if "none" in data["condition"].lower():
                                follow_through_data_list.append(np.array([self.robot.get_joint_index(data["target"]), self.robot.get_joint_index(data["source"]), data["inverse"]]))
                            else:
                                follow_through_data_list.append(np.array([self.robot.get_joint_index(data["target"]), self.robot.get_joint_index(data["source"]), data["inverse"], self.robot.get_joint_index(data["condition"]), data["lower_limit"], data["upper_limit"]]))
                        block.dmp.set_principle_parameters(p_ant=animation["Anticipation"], p_follow=animation["Follow_Through"], p_arc=animation["Arcs"], p_slow=animation["Slow_In_Slow_Out"], p_progression=["fast", "fast", "fast"], p_time=animation["Timing"], p_exa=animation["Exaggeration"], p_follow_data=follow_through_data_list)
                    self.timeline.append_block(block)                

                logging.info("[LLAGCore] New plan added to timeline.")

            await asyncio.sleep(0.1)

    def set_plan(self, mapped_animation_sequence):
        if len(self.timeline.plan) > 0:
            logging.info("[LLAGCore] Clearing existing plan.")
            self.timeline.plan = []

        
        
        for animation in mapped_animation_sequence:
            block = LLAGTimelineBlock(animation["motion_primitive"], primitive_path=self.primitive_lib_path, idle_data_yaml_path=self.robot_description_path)
            self.experiment_plan.append(deepcopy(block))
            follow_through_data_list = []
            for key in animation["Follow_Through_Data"].keys():
                data = animation["Follow_Through_Data"][key]
                if "none" in data["condition"].lower():
                    follow_through_data_list.append(np.array([self.robot.get_joint_index(data["target"]), self.robot.get_joint_index(data["source"]), data["inverse"]]))
                else:
                    follow_through_data_list.append(np.array([self.robot.get_joint_index(data["target"]), self.robot.get_joint_index(data["source"]), data["inverse"], self.robot.get_joint_index(data["condition"]), data["lower_limit"], data["upper_limit"]]))
            block.dmp.set_principle_parameters(p_ant=animation["Anticipation"], p_follow=animation["Follow_Through"], p_arc=animation["Arcs"], p_slow=animation["Slow_In_Slow_Out"], p_progression=["fast", "fast", "fast"], p_time=animation["Timing"], p_exa=animation["Exaggeration"], p_follow_data=follow_through_data_list)
            self.timeline.append_block(block)
        if self.timeline.current_block.name_identifier.lower().startswith("zero"):
            self.timeline.advance_block()

    async def rt_data_handler(self):
        # Define 4 positions: (yaw, pitch) in radians
        yaw_angle = np.radians(30)   # ±30°
        pitch_angle = np.radians(10) # ±20°

        positions = [
            ( yaw_angle, -pitch_angle),  # bottom-right
            ( yaw_angle,  pitch_angle),  # top-right
            ( yaw_angle/2,  pitch_angle),  # top-right
            ( 0,  pitch_angle),  # top-right    
            ( 0,  -pitch_angle),  # top-right   
            (-yaw_angle/2, -pitch_angle),  # bottom-left        
            (-yaw_angle, -pitch_angle),  # bottom-left
            
            (-yaw_angle,  pitch_angle),  # top-left
            (0,  pitch_angle),  # top-left
        ]
        
            

        hold_duration = 2.0  # seconds to stay at each position
        dt = 0.05
        steps_per_position = int(hold_duration / dt)

        position_index = 0
        step_counter = 0
        experiment_steps = 0

        while True:            
            
            human_yaw, human_pitch = positions[position_index]

            # Get camera orientation
            camera_yaw = deepcopy(self.timeline.current_block.dmp.y[3])
            camera_pitch = deepcopy(self.timeline.current_block.dmp.y[4])

            # Compute offsets (what the camera "sees")
            offset_yaw = human_yaw - camera_yaw
            offset_pitch = human_pitch - camera_pitch
            # Save to timeline
            self.timeline.rt_data = {
                'x': offset_yaw,
                'y': offset_pitch,
            }

            # Step to next position after hold_duration
            step_counter += 1
            if step_counter >= steps_per_position:
                step_counter = 0
                position_index = (position_index + 1) % len(positions)

            await asyncio.sleep(dt)

    async def _safe_task(self, coro, task_name: str):
        """Wrapper to catch and log exceptions in async tasks"""
        try:
            await coro
        except Exception as e:
            logging.error(f"Exception in {task_name}: {e}", exc_info=True)
            raise

    async def run(self):
        # Session setup
        if self.use_real_robot:
            import peppertoolbox
            self.session = peppertoolbox.PepperSession(PEPPER_IP="129.69.223.125")
            self.pepper_audio_service = self.session.session.service("ALAudioDevice")
            self.session.posture_service.goToPosture("StandInit", 0.1)
        if self.use_virtual_robot:
            self.virtual_session = VirtualSession(urdf_path=self.urdf_path)
            self.joint_names_list = self.virtual_session.get_joint_names()
            self.markdown_text = self.virtual_session.server.gui.add_markdown(content=f"Timeline Plan:\n {self.timeline.plan}")

            text_field = self.virtual_session.server.gui.add_text(label="Context Input", initial_value="")
            button = self.virtual_session.server.gui.add_button(label="Send Context")

            @button.on_click
            def _(_):
                context = text_field.value.strip()
                self.context_store.handle_context_input({"type": "text", "context": str(context)})
                text_field.value = ""

            #if not self.use_real_robot:
            while True:
                clients = self.virtual_session.server.get_clients()
                if clients:
                    print(f"Connected clients: {len(clients)}")
                    self.viser_ready = True
                    break
                else:
                    print("Waiting for clients to connect...")
                    await asyncio.sleep(0.5)

        # Wrap tasks to show which one fails
        await asyncio.gather(
            self._safe_task(self.controller_loop(), "controller_loop"),
            self._safe_task(self.context_listener(), "context_listener"),
            self._safe_task(self.planner_trigger_loop(), "planner_trigger_loop"),
            #self._safe_task(self.rt_data_handler(), "rt_data_handler"),
        )