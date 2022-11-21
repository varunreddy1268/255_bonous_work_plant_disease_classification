import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Markdown
from IPython import display
from openvino.runtime import Core
model_path = Path("/Users/varunreddyseelam/Downloads/O-MyNatural/models/model.pb")
ir_path = Path(model_path).with_suffix(".xml")
mo_command = f"""mo
                 --input_model '{model_path}'
                 --input_shape '[1,224,224,3]'
                 --mean_values= '[127.5,127.5,127.5]'
                 --scale_values= ''[127.5]'
                 --data_type FP16
                 --output_dir '{model_path.parent}'
                 """
mo_command = " ".join(mo_command.split())
print("Model Optimizer command to convert TensorFlow to OpenVINO:")
#display(Markdown(f"`{mo_command}`"))
if not ir_path.exists():
    print("Exporting TensorFlow model to IR... This may take a few minutes.")
    exec(mo_command)
else:
    print(f"IR model {ir_path} already exists.")
