import re
import yaml
import pybullet_data
import datetime
from typing import Literal
from pathlib import Path

DIR_ROOT = Path(__file__).parent.parent.resolve(True)
DIR_PYBULLET_DATA = Path(pybullet_data.getDataPath()).resolve(True)
DIR_VIDEOS = (DIR_ROOT / "videos").resolve(False)  # Might not exist yet


def get_project_filepath(filename: str) -> str:
    """
    Obtain the absolute path pointing towards a filename relative to
    the project's root directory.

    Args:
        filename (str): Filename relative to the project root

    Returns:
        str: Absolute path towards filename, if it exists.
    """
    return str((DIR_ROOT / filename).resolve(True))


def get_pybullet_filepath(filename: str) -> str:
    """
    Obtain the absolute path pointing towards a filename relative to
    pybullet's data directory

    Args:
        filename (str): Filename relative to pybullet's data directory

    Returns:
        str: Absolute path towards filename, if it exists.
    """
    return str((DIR_PYBULLET_DATA / filename).resolve(True))


def generate_video_filename() -> str:
    """
    Generate an absolute filename for video output, timestamped to current time.

    Returns:
        str: Timestamped absolute path to store the video at.
    """
    # Create the video directory if it does not exist
    if not DIR_VIDEOS.exists():
        DIR_VIDEOS.mkdir()
    filename = datetime.datetime.now().strftime("video_%Y%m%d_%H%M%S.MP4")
    return str(DIR_VIDEOS / filename)


def load_yaml(filename: str) -> dict:
    """
    Load a YAML file into a python dictionary.

    Args:
        filename (str): Absolute path to YAML file

    Returns:
        dict: YAML in python dictionary format
    """
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def match_joint_type(
    joint_name: str, joint_type: Literal["chassis", "hip", "thigh", "calf", "foot"]
) -> bool:
    """
    Check whether the joint with given joint_name matches the type of joint joint_type.

    Args:
        joint_name (str): Name of the joint from the URDF
        joint_type (str): Type of joint

    Returns:
        bool: Whether the name matches the joint type
    """
    if joint_type == "chassis":
        pattern = re.compile("\w*floating_base\w*")
    else:
        pattern = re.compile(f"\w+_{joint_type}_\w+")

    return bool(pattern.match(joint_name))
