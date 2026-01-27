from llag_block import LLAGTimelineBlock
import logging
from copy import deepcopy
import numpy as np

class LLAGTimeline:
    def __init__(self, primitive_path=None, robot_description_path=None):
        """Initialize timeline with optional robot-specific paths for idle block.
        
        Args:
            primitive_path: Path to robot primitives folder
            robot_description_path: Path to robot YAML configuration
        """
        self.plan = []
        self.primitive_path = primitive_path
        self.robot_description_path = robot_description_path
        
        logging.info("Initializing timeline with random idle block")
        if primitive_path and robot_description_path:
            self.current_block = LLAGTimelineBlock("idle", primitive_path=primitive_path, idle_data_yaml_path=robot_description_path)
        else:
            self.current_block = None  # Will be set when first block is added

        # initalize real time data
        self.rt_data = {"x": 0.0, "y": 0.0}

    def update_plan(self, new_plan):
        #logging.info("update_plan")
        #logging.info(new_plan)
        self.plan = new_plan
        if not self.current_block and self.plan:
            #logging.info("directly setting current block")
            self.current_block = self.plan.pop(0)

    def append_block(self, new_block):
        self.plan.append(new_block)

        if not self.current_block and self.plan:
            #logging.info("directly setting current block")
            self.current_block = self.plan.pop(0)

    def get_current_block(self):
        return self.current_block

    def advance_block(self):
        last_block_state = None
        
        #temporarily save state of last current block
        if self.current_block:
            last_block_state = deepcopy(self.current_block.dmp.get_state())
            last_block_goal = self.current_block.dmp.goal
        
        if self.plan:
            self.current_block = self.plan.pop(0)
            
            # Manual adaptation of gesture primitives - much easier to change them here instead of manually updating the root data file of the gestures.
            if self.current_block.name_identifier.lower().startswith("sway") or self.current_block.name_identifier.lower().startswith("indicate_right") or self.current_block.name_identifier.lower().startswith("bow"):
                exa_array = np.ones(self.current_block.dmp.n_dim) * self.current_block.dmp.modulation.p_exa
                exa_array[3] = 0.4
                exa_array[4] = 0.4
                self.current_block.dmp.set_principle_parameters(p_exa = exa_array)
            if self.current_block.name_identifier.lower().startswith("celebrate"):
                self.current_block.dmp.set_principle_parameters(p_time=1.0, p_progression=["fast", "fast", "slow"])
            if self.current_block.name_identifier.lower().startswith("point"):
                exa_array = np.ones(self.current_block.dmp.n_dim) * self.current_block.dmp.modulation.p_exa
                exa_array[3] = 0.25
                exa_array[4] = 0.25
                self.current_block.dmp.set_principle_parameters(p_exa = exa_array)
            if self.current_block.name_identifier.lower().startswith("wave"):
                self.current_block.dmp.set_principle_parameters(p_slow=False, p_progression=["fast", "slow", "moderate"])

            logging.info(f"[timeline] Advancing to next block; plan still has items; next block: {self.current_block.name_identifier}; remaining plan length: {len(self.plan)}; remaining plan: {[block.name_identifier for block in self.plan]}")

        else:            
            self.current_block = LLAGTimelineBlock("idle", primitive_path=self.primitive_path, idle_data_yaml_path=self.robot_description_path)

            logging.info(f"[timeline] Advancing to next block; plan is empty; inserting idle block; next block: {self.current_block.name_identifier}")

            # modulate motion randomization idle primitive (zero motion trajectory)
            if self.current_block.name_identifier.lower().startswith("zero"):
                
                self.current_block.dmp.set_principle_parameters(p_rand=400)
                self.current_block.dmp.forcing_term.w_original = self.current_block.dmp.forcing_term.w
                self.current_block.dmp.set_principle_parameters(p_arc=40)
                exa_array = np.ones(self.current_block.dmp.n_dim)
                exa_array[0] = 0.3
                exa_array[1] = 0.3
                exa_array[2] = 0.4
                exa_array[3] = 0.2
                exa_array[4] = 0.2
                self.current_block.dmp.set_principle_parameters(p_exa = exa_array)
            
        # seamlessly join primitives and keep head pose of pepper
        if last_block_state:
            self.current_block.dmp.y = last_block_state["y"]
            self.current_block.dmp.yd = last_block_state["yd"]
            self.current_block.dmp.ydd = last_block_state["ydd"]
            self.current_block.dmp.goal[3] = last_block_goal[3]
            self.current_block.dmp.goal[4] = last_block_goal[4]
        self.current_block.update_goal(self.rt_data)