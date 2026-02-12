# Robot Template

Use this template to add a new robot to the system.

## Quick Start

1. **Copy** this directory:
   ```bash
   cp -r robot_template robot_{name}
   ```

2. **Rename** all files:
   ```
   robot_{name}/
   ├── robot_{name}.py
   ├── robot_{name}.yaml
   ├── robot_{name}_description/    ← place your URDF (as robot_template.urdf)
   ├── robot_{name}_primitives/     ← place DMP saves here
   ```

   For the Animation DMP saves refer to the animation-dmp repository. Note that the saves need to be named such that names align with primitives listen in the robot_template.yaml file!

3. **Edit `robot_{name}.py`**:
   - Rename the class (e.g. `MyRobot`).
   - Fill in `JOINT_NAMES` and `DEFAULT_JOINT_ANGLES`.
   - Update `execute_state_on_virtual_robot()` to extract and reorder joints.
   - Update the `create_robot()` factory function.

4. **Edit `robot_{name}.yaml`**:
   - Write `capabilities` (free text for the LLM).
   - List all motion primitives in `primitive_lib`.
   - Set `idle_lib` to at least one resting primitive.
   - Tune `parameter_ranges` for your robot's joint limits.
   - `Follow_Through_Data` is auto-computed from the URDF on first load and
     written back to the YAML. No manual entry needed (but you can override
     it by providing the key yourself).

5. **Add assets**:
   - Place your URDF and meshes in `robot_{name}_description/`.
   - Record trajectories, convert them to DMPs, and place the `.json` + `_weights.npy` files in `robot_{name}_primitives/`.

6. **Activate** in `config.yaml`:
   ```yaml
   robot: {name}
   ```
