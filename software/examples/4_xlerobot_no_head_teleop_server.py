# To Run on the host
'''python
PYTHONPATH=src python -m lerobot.robots.xlerobot_no_head.xlerobot_no_head_host --robot.id=my_xlerobot_no_head_pc
'''

# To Run the websocket teleop:
'''python
PYTHONPATH=src python -m examples.4_xlerobot_no_head_ws
'''

import time
import threading
import math
import numpy as np
from flask import Flask, request, jsonify

from lerobot.robots.xlerobot_no_head import XLerobotNoHeadClient, XLerobotNoHeadClientConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.model.SO101Robot import SO101Kinematics


# Keymaps (semantic action: key)
LEFT_KEYMAP = {
    'shoulder_pan+': 'q', 'shoulder_pan-': 'e',   # 大臂旋转
    'wrist_roll+': 'r', 'wrist_roll-': 'f',       # 手腕旋转
    'gripper+': 't', 'gripper-': 'g',             # 夹爪
    'x+': 'w', 'x-': 's',                         # 大臂下压上抬
    'y+': 'a', 'y-': 'd',                         # 二臂下压上抬
    'pitch+': 'z', 'pitch-': 'x',                 # 手腕下压上抬
    'reset': 'c',
    # For head motors
    "head_motor_1+": "<", "head_motor_1-": ">",
    "head_motor_2+": ",", "head_motor_2-": ".",
    
    'triangle': 'y',  # Rectangle trajectory key
}
RIGHT_KEYMAP = {
    'shoulder_pan+': '7', 'shoulder_pan-': '9',   # 大臂旋转
    'wrist_roll+': '/', 'wrist_roll-': '*',       # 手腕旋转
    'gripper+': '+', 'gripper-': '-',             # 夹爪
    'x+': '8', 'x-': '2',                         # 大臂下压上抬
    'y+': '4', 'y-': '6',                         # 二臂下压上抬
    'pitch+': '1', 'pitch-': '3',                 # 手腕下压上抬
    'reset': '0',

    'triangle': 'Y',  # Rectangle trajectory key
}

LEFT_JOINT_MAP = {
    "shoulder_pan": "left_arm_shoulder_pan",
    "shoulder_lift": "left_arm_shoulder_lift",
    "elbow_flex": "left_arm_elbow_flex",
    "wrist_flex": "left_arm_wrist_flex",
    "wrist_roll": "left_arm_wrist_roll",
    "gripper": "left_arm_gripper",
}
RIGHT_JOINT_MAP = {
    "shoulder_pan": "right_arm_shoulder_pan",
    "shoulder_lift": "right_arm_shoulder_lift",
    "elbow_flex": "right_arm_elbow_flex",
    "wrist_flex": "right_arm_wrist_flex",
    "wrist_roll": "right_arm_wrist_roll",
    "gripper": "right_arm_gripper",
}

# Head motor mapping
HEAD_MOTOR_MAP = {
    "head_motor_1": "head_motor_1",
    "head_motor_2": "head_motor_2",
}

class RectangularTrajectory:
    """
    Generates a rectangular trajectory on the x-y plane with sinusoidal velocity profiles.
    The rectangle is divided into 4 line segments, each with smooth acceleration/deceleration.
    """
    def __init__(self, width=0.06, height=0.06, segment_duration=0.91):
        """
        Initialize rectangular trajectory parameters.
        
        Args:
            width: Rectangle width in meters
            height: Rectangle height in meters  
            segment_duration: Time for each line segment in seconds
        """
        self.width = width
        self.height = height
        self.segment_duration = segment_duration
        self.total_duration = 4 * segment_duration
        
    def get_trajectory_point(self, current_x, current_y, t):
        """
        Get the target x, y position at time t for the rectangular trajectory.
        
        Args:
            current_x: Starting x position
            current_y: Starting y position
            t: Time since trajectory start (0 to total_duration)
            
        Returns:
            tuple: (target_x, target_y)
        """
        # Determine which segment we're in
        segment = int(t / self.segment_duration)
        segment_t = t % self.segment_duration
        
        # Normalize segment time (0 to 1)
        normalized_t = segment_t / self.segment_duration
        
        # Sinusoidal velocity profile: smooth acceleration and deceleration
        # s(t) = 0.5 * (1 - cos(π * t)) gives smooth 0 to 1 transition
        smooth_t = 0.5 * (1 - math.cos(math.pi * normalized_t))
        
        # Define rectangle corners relative to starting position
        corners = [
            (current_x, current_y),                           # Start (bottom-left)
            (current_x + self.width, current_y),              # Bottom-right
            (current_x + self.width, current_y + self.height), # Top-right  
            (current_x, current_y + self.height),             # Top-left
            (current_x, current_y)                            # Back to start
        ]
        
        # Clamp segment to valid range
        segment = max(0, min(3, segment))
        
        # Interpolate between current corner and next corner
        start_corner = corners[segment]
        end_corner = corners[segment + 1]
        
        target_x = start_corner[0] + smooth_t * (end_corner[0] - start_corner[0])
        target_y = start_corner[1] + smooth_t * (end_corner[1] - start_corner[1])
        
        return target_x, target_y


class SimpleTeleopArm:
    def __init__(self, kinematics, joint_map, initial_obs, prefix="left", kp=0.41):
        self.kinematics = kinematics
        self.joint_map = joint_map
        self.prefix = prefix  # To distinguish left and right arm
        self.kp = kp
        # Initial joint positions
        self.joint_positions = {
            "shoulder_pan": initial_obs[f"{prefix}_arm_shoulder_pan.pos"],
            "shoulder_lift": initial_obs[f"{prefix}_arm_shoulder_lift.pos"],
            "elbow_flex": initial_obs[f"{prefix}_arm_elbow_flex.pos"],
            "wrist_flex": initial_obs[f"{prefix}_arm_wrist_flex.pos"],
            "wrist_roll": initial_obs[f"{prefix}_arm_wrist_roll.pos"],
            "gripper": initial_obs[f"{prefix}_arm_gripper.pos"],
        }
        # Set initial x/y to fixed values
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        # Set the degree step and xy step
        self.degree_step = 0.3
        self.xy_step = 0.00081
        # Set target positions to zero for P control
        self.target_positions = {
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        }
        self.zero_pos = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }

        # Rectangular trajectory instance
        self.rectangular_trajectory = RectangularTrajectory(
            width=0.06,          # 6cm wide rectangle
            height=0.06,         # 4cm tall rectangle  
            segment_duration=1.01 # 3 seconds per line segment
        )

    def move_to_zero_position(self, robot):
        print(f"[{self.prefix}] Moving to Zero Position: {self.zero_pos} ......")
        self.target_positions = self.zero_pos.copy()  # Use copy to avoid reference issues
        
        # Reset kinematic variables to their initial state
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        # Don't let handle_keys recalculate wrist_flex - set it explicitly
        self.target_positions["wrist_flex"] = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)
    
    def move_to_hang_position(self, robot):
        if self.prefix == "left":
            hang_pos = {
                'shoulder_pan': 41.5,
                'shoulder_lift': -93.1,
                'elbow_flex': 82.2,
                'wrist_flex': 38.9,
                'wrist_roll': -52.5,
                'gripper': 0.0
            }
        else:
            hang_pos = {
                'shoulder_pan': -52.8,
                'shoulder_lift': -91.3,
                'elbow_flex': 83.0,
                'wrist_flex': 47.9,
                'wrist_roll': -56.6,
                'gripper': 0.0
            }
        print(f"[{self.prefix}] Moving to Hang Position: {hang_pos} ......")
        self.target_positions = hang_pos.copy()  # Use copy to avoid reference issues
        
        # Update kinematic variables to match hang position
        # Approximate x, y based on hang position (these values may need tuning)
        self.current_x = 0.1629
        self.current_y = 0.1131
        self.pitch = 0.0
        
        action = self.p_control_action(robot)
        robot.send_action(action)

    def execute_rectangular_trajectory(self, robot, fps=30):
        """
        Execute a blocking rectangular trajectory on the x-y plane.
        
        Args:
            robot: Robot instance to send actions to
            fps: Control loop frequency
        """
        print(f"[{self.prefix}] Starting rectangular trajectory...")
        print(f"[{self.prefix}] Rectangle: {self.rectangular_trajectory.width:.3f}m x {self.rectangular_trajectory.height:.3f}m")
        print(f"[{self.prefix}] Duration: {self.rectangular_trajectory.total_duration:.3f}s total")
        
        # Store starting position
        start_x = self.current_x
        start_y = self.current_y
        
        # Execute trajectory
        start_time = time.time()
        dt = 1.0 / fps
        
        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Check if trajectory is complete
            if elapsed_time >= self.rectangular_trajectory.total_duration:
                print(f"[{self.prefix}] Rectangular trajectory completed!")
                break
                
            # Get target position from trajectory
            target_x, target_y = self.rectangular_trajectory.get_trajectory_point(
                start_x, start_y, elapsed_time
            )
            
            # Update current position
            self.current_x = target_x
            self.current_y = target_y
            
            # Calculate inverse kinematics
            try:
                joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
                self.target_positions["shoulder_lift"] = joint2
                self.target_positions["elbow_flex"] = joint3
                
                # Update wrist_flex coupling
                self.target_positions["wrist_flex"] = (
                    -self.target_positions["shoulder_lift"]
                    -self.target_positions["elbow_flex"]
                    + self.pitch
                )
                
                # Get action
                action = self.p_control_action(robot)
                
                # Determine which arm is executing and send appropriate action structure
                if self.prefix == "left":
                    # Send left arm action with empty actions for other components
                    robot_action = {**action, **{}, **{}, **{}}
                elif self.prefix == "right":
                    # Send right arm action with empty actions for other components
                    robot_action = {**{}, **action, **{}, **{}}
                
                # Send action to robot
                robot.send_action(robot_action)
                
                # Get observation and log data
                obs = robot.get_observation()
                # log_rerun_data(obs, robot_action)
                
            except Exception as e:
                print(f"[{self.prefix}] IK failed at x={self.current_x:.4f}, y={self.current_y:.4f}: {e}")
                break
                
            # Maintain control frequency
            # busy_wait(dt)
        
        print(f"[{self.prefix}] Trajectory execution finished.")

    def handle_keys(self, key_state):
        # Joint increments
        if key_state.get('shoulder_pan+'):
            self.target_positions["shoulder_pan"] += self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('shoulder_pan-'):
            self.target_positions["shoulder_pan"] -= self.degree_step
            print(f"[{self.prefix}] shoulder_pan: {self.target_positions['shoulder_pan']}")
        if key_state.get('wrist_roll+'):
            self.target_positions["wrist_roll"] += self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('wrist_roll-'):
            self.target_positions["wrist_roll"] -= self.degree_step
            print(f"[{self.prefix}] wrist_roll: {self.target_positions['wrist_roll']}")
        if key_state.get('gripper+'):
            self.target_positions["gripper"] += self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('gripper-'):
            self.target_positions["gripper"] -= self.degree_step
            print(f"[{self.prefix}] gripper: {self.target_positions['gripper']}")
        if key_state.get('pitch+'):
            self.pitch += self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")
        if key_state.get('pitch-'):
            self.pitch -= self.degree_step
            print(f"[{self.prefix}] pitch: {self.pitch}")

        # XY plane (IK)
        moved = False
        if key_state.get('x+'):
            self.current_x += self.xy_step
            moved = True
            print(f"[{self.prefix}] x+: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('x-'):
            self.current_x -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x-: {self.current_x:.4f}, y: {self.current_y:.4f}")
        if key_state.get('y+'):
            self.current_y += self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y+: {self.current_y:.4f}")
        if key_state.get('y-'):
            self.current_y -= self.xy_step
            moved = True
            print(f"[{self.prefix}] x: {self.current_x:.4f}, y-: {self.current_y:.4f}")
        if moved:
            joint2, joint3 = self.kinematics.inverse_kinematics(self.current_x, self.current_y)
            self.target_positions["shoulder_lift"] = joint2
            self.target_positions["elbow_flex"] = joint3
            print(f"[{self.prefix}] shoulder_lift: {joint2}, elbow_flex: {joint3}")

        # Wrist flex is always coupled to pitch and the other two
        self.target_positions["wrist_flex"] = (
            -self.target_positions["shoulder_lift"]
            -self.target_positions["elbow_flex"]
            + self.pitch
        )
        # print(f"[{self.prefix}] wrist_flex: {self.target_positions['wrist_flex']}")

    def p_control_action(self, robot):
        obs = robot.get_observation()
        current = {j: obs[f"{self.prefix}_arm_{j}.pos"] for j in self.joint_map}
        action = {}
        for j in self.target_positions:
            error = self.target_positions[j] - current[j]
            control = self.kp * error
            action[f"{self.joint_map[j]}.pos"] = current[j] + control
        return action


class HTTPTeleopServer:
    """Simple HTTP server (Flask) for robot teleoperation control.

    Exposes POST /actions which accepts a plain text sequence like 'qqq' or
    'qrg'. Each character is interpreted as a single key press. Action
    characters (those present in `char_to_action`) map to arm actions; base
    keys map to base movements.
    """
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host
        self.port = port
        self.active_action_chars = set()
        self.active_base_keys = set()
        self.lock = threading.Lock()
        # reverse map from character to action name (filled after maps available)
        self.char_to_action = {}

        # References to teleop arms and robot (registered by main)
        self.left_arm = None
        self.right_arm = None
        self.robot = None

        # allowed base keys (add 'u' and 'o' for left/right turn)
        self.base_keys = {"i", "j", "k", "l", "u", "o"}

        # Flask app
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self):
        @self.app.route("/actions", methods=["POST", "GET"])
        def actions():
            # Accept several formats: raw body text, form field 'actions', or
            # JSON {"actions": "qqq"}. For GET, accept query param 'q'.
            seq = None
            if request.method == 'GET':
                seq = request.args.get('q', '')
            else:
                # Try JSON
                if request.is_json:
                    data = request.get_json(silent=True) or {}
                    seq = data.get('actions') or data.get('q')
                # Try form
                if not seq:
                    seq = request.form.get('actions') or request.form.get('q')
                # Fallback to raw body
                if not seq:
                    seq = request.get_data(as_text=True)

            seq = (seq or '').strip()
            if not seq:
                return jsonify({'error': 'empty actions'}), 400

            allowed_chars = set(self.char_to_action.keys()) | self.base_keys
            for ch in seq:
                if ch not in allowed_chars:
                    return jsonify({'error': f"invalid character '{ch}'"}), 400

            # Run the sequence in a background thread to avoid blocking Flask
            t = threading.Thread(target=self._run_sequence, args=(seq,), daemon=True)
            t.start()
            return jsonify({'status': 'ok', 'received': seq})

        @self.app.route("/hang", methods=["POST", "GET"])
        def hang():
            """Move one or both arms to the 'hang' position.

            Query params / JSON / form:
              arm=left|right|both  (default: both)
            """
            arm = None
            if request.method == 'GET':
                arm = request.args.get('arm', 'both')
            else:
                if request.is_json:
                    data = request.get_json(silent=True) or {}
                    arm = data.get('arm')
                if not arm:
                    arm = request.form.get('arm')
                if not arm:
                    arm = request.get_data(as_text=True).strip() or 'both'

            arm = (arm or 'both').lower()
            if arm not in ('left', 'right', 'both'):
                return jsonify({'error': "invalid arm, must be 'left','right',or 'both'"}), 400

            # Handler must have been registered by main()
            if self.robot is None or (self.left_arm is None and self.right_arm is None):
                return jsonify({'error': 'teleop arms not registered on server'}), 500

            def _do_hang():
                try:
                    if arm in ('left', 'both') and self.left_arm is not None:
                        self.left_arm.move_to_hang_position(self.robot)
                    if arm in ('right', 'both') and self.right_arm is not None:
                        self.right_arm.move_to_hang_position(self.robot)
                except Exception as e:
                    print(f"[HTTP] Error executing hang: {e}")

            threading.Thread(target=_do_hang, daemon=True).start()
            return jsonify({'status': 'ok', 'arm': arm})

    def _run_sequence(self, seq, hold=0.08, gap=0.02):
        """Execute a sequence of characters sequentially in background."""
        for ch in seq:
            with self.lock:
                if ch in self.char_to_action:
                    self.active_action_chars.add(ch)
                else:
                    self.active_base_keys.add(ch)
            time.sleep(hold)
            with self.lock:
                if ch in self.char_to_action:
                    self.active_action_chars.discard(ch)
                else:
                    self.active_base_keys.discard(ch)
            time.sleep(gap)

    def get_active_keys(self):
        """Map active characters to semantic action names and return snapshot."""
        with self.lock:
            return set(self.char_to_action[ch] for ch in self.active_action_chars if ch in self.char_to_action)

    def get_active_action_chars(self):
        """Return a snapshot set of raw active action characters (e.g. 'q','7').

        Use this in the main loop to distinguish left vs right arm mappings
        (LEFT_KEYMAP/RIGHT_KEYMAP map semantic action names to distinct
        characters). Returning raw characters preserves which side the
        character belonged to.
        """
        with self.lock:
            return set(self.active_action_chars)

    def get_active_base_keys(self):
        with self.lock:
            return set(self.active_base_keys)

    def clear_action(self, action_name):
        with self.lock:
            for c, a in list(self.char_to_action.items()):
                if a == action_name:
                    self.active_action_chars.discard(c)

    def register_teleop_bindings(self, left_arm=None, right_arm=None, robot=None):
        """Register left/right SimpleTeleopArm instances and the robot instance
        so HTTP endpoints can trigger blocking arm operations (like hang).
        """
        with self.lock:
            self.left_arm = left_arm
            self.right_arm = right_arm
            self.robot = robot

    def start(self):
        print(f"[HTTP] Starting Flask server on http://{self.host}:{self.port}")
        # Run Flask in a daemon thread so main loop can continue
        t = threading.Thread(target=self.app.run, kwargs={'host': self.host, 'port': self.port, 'threaded': True}, daemon=True)
        t.start()


# Previously used to run an asyncio TCP server. Replaced by the Flask HTTP
# server (HTTPTeleopServer) below.


def main():
    # Teleop parameters
    FPS = 50
    ip = "192.168.3.150"  # This is for zmq connection
    robot_name = "my_xlerobot_no_head_pc"

    # WebSocket server configuration
    WS_HOST = "0.0.0.0"
    WS_PORT = 8765

    # For zmq connection
    robot_config = XLerobotNoHeadClientConfig(remote_ip=ip, id=robot_name)
    robot = XLerobotNoHeadClient(robot_config)    

    try:
        robot.connect()
        print(f"[MAIN] Successfully connected to robot")
    except Exception as e:
        print(f"[MAIN] Failed to connect to robot: {e}")
        print(robot_config)
        print(robot)
        return

    # init_rerun(session_name="xlerobot_no_head_teleop_ws")

    # Start HTTP (Flask) server in a background thread
    http_server = HTTPTeleopServer(host=WS_HOST, port=WS_PORT)
    # Populate reverse mapping from keyboard char -> action name for shorthand parsing
    # Build char->action map: prefer LEFT_KEYMAP when duplicate characters exist
    char_map = {}
    # Normalize keys (strip whitespace) when building the reverse map.
    for action, key in LEFT_KEYMAP.items():
        k = str(key).strip()
        char_map[k] = action
    for action, key in RIGHT_KEYMAP.items():
        k = str(key).strip()
        if k not in char_map:
            char_map[k] = action
    http_server.char_to_action = char_map
    http_server.start()
    # Give the server a moment to start
    time.sleep(1)
    print(f"[MAIN] HTTP server started on http://{WS_HOST}:{WS_PORT}")
    print(f"[MAIN] Clients can POST plain character sequences to /actions, e.g. 'qqq' or 'qrg'.")

    # Init the arm instances
    obs = robot.get_observation()
    kin_left = SO101Kinematics()
    kin_right = SO101Kinematics()
    left_arm = SimpleTeleopArm(kin_left, LEFT_JOINT_MAP, obs, prefix="left")
    right_arm = SimpleTeleopArm(kin_right, RIGHT_JOINT_MAP, obs, prefix="right")

    # Register arms and robot with HTTP server so endpoints can trigger blocking ops
    http_server.register_teleop_bindings(left_arm=left_arm, right_arm=right_arm, robot=robot)

    # Move both arms to zero position at start
    left_arm.move_to_zero_position(robot)
    right_arm.move_to_zero_position(robot)

    try:
        while True:
            # Get active characters and base keys from HTTP server
            active_chars = http_server.get_active_action_chars()
            active_base_keys = http_server.get_active_base_keys()

            # Build key states for left and right arms by checking the
            # corresponding character (LEFT_KEYMAP/RIGHT_KEYMAP map semantic
            # action names -> character). This preserves left/right
            # separation: e.g. LEFT 'shoulder_pan+' maps to 'q', RIGHT maps to '7'.
            left_key_state = {action: (LEFT_KEYMAP[action] in active_chars) for action in LEFT_KEYMAP.keys()}
            right_key_state = {action: (RIGHT_KEYMAP[action] in active_chars) for action in RIGHT_KEYMAP.keys()}

            # Handle rectangular trajectory for left arm (triangle key)
            if left_key_state.get('triangle'):
                print("[MAIN] Left arm rectangular trajectory triggered!")
                left_arm.execute_rectangular_trajectory(robot, fps=FPS)
                # Clear the triangle key after execution
                http_server.clear_action('triangle')
                continue

            # Handle rectangular trajectory for right arm (triangle key)  
            if right_key_state.get('triangle'):
                print("[MAIN] Right arm rectangular trajectory triggered!")
                right_arm.execute_rectangular_trajectory(robot, fps=FPS)
                # Clear the triangle key after execution
                http_server.clear_action('triangle')
                continue

            # Handle reset for left arm
            if left_key_state.get('reset'):
                left_arm.move_to_zero_position(robot)
                # Clear the reset key after execution
                http_server.clear_action('reset')
                continue  

            # Handle reset for right arm
            if right_key_state.get('reset'):
                right_arm.move_to_zero_position(robot)
                # Clear the reset key after execution
                http_server.clear_action('reset')
                continue

            # Handle normal teleoperation
            left_arm.handle_keys(left_key_state)
            right_arm.handle_keys(right_key_state)

            left_action = left_arm.p_control_action(robot)
            right_action = right_arm.p_control_action(robot)

            # Base action (convert active base keyboard chars to base movement commands)
            keyboard_keys = np.array(list(active_base_keys))
            base_action = robot._from_keyboard_to_base_action(keyboard_keys) or {}

            # Combine actions
            action = {**left_action, **right_action, **base_action}
            robot.send_action(action)

            obs = robot.get_observation()
            # print(f"[MAIN] Observation: {obs}")
            # log_rerun_data(obs, action)
            # busy_wait(1.0 / FPS)
            
    except KeyboardInterrupt:
        print("[MAIN] Keyboard interrupt received")
    finally:
        robot.disconnect()
        print("Teleoperation ended.")


if __name__ == "__main__":
    main()
