# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import platform


# START: [FIX 1] Set correct working directory to project root
# This ensures that all relative paths in configs and arguments work correctly,
# regardless of where the script is called from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(project_root)
# END: [FIX 1]

# START: [FIX] Add DLL paths to environment to prevent WinError 127 on torch/cudnn import
# This logic is copied from train.py to be more robust.
env_path = os.environ.get("CONDA_PREFIX", r"C:\Users\ADMIN\anaconda3\envs\torch310")

cuda_version = "v11.8" # This version is from your train.py, check if it's correct for your system
cuda_path = os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"), "NVIDIA GPU Computing Toolkit", "CUDA", cuda_version)

paths_to_add = [
    os.getcwd(), # Add current working dir
    os.path.join(cuda_path, "bin"),
    os.path.join(env_path, r"Lib\site-packages\nvidia\cudnn\bin"),
    os.path.join(env_path, r"Lib\site-packages\nvidia\cublas\bin"),
    os.path.join(env_path, r"Library\bin")
]

valid_paths = [p for p in paths_to_add if os.path.exists(p)]
if valid_paths:
    os.environ["PATH"] = ";".join(valid_paths) + ";" + os.environ.get("PATH", "")
    if platform.system() == "Windows" and sys.version_info >= (3, 8):
        for p in valid_paths:
            if os.path.exists(p):
                os.add_dll_directory(p)
# END: [FIX]
os.environ["FLAGS_enable_pir_api"] = "0"

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import argparse

from tools.program import load_config, merge_config, ArgsParser
from ppocr.utils.export_model import export


def main():
    FLAGS = ArgsParser().parse_args()
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    # export model
    export(config)


if __name__ == "__main__":
    main()
