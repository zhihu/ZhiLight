# coding=utf-8
#
import os
import time

from common_args import *
from zhilight.dynamic_batch import DynamicBatchConfig, GeneratorArg, DynamicBatchGenerator
from zhilight.models.auto_model import AutoModel
from zhilight.models.deepseek_vl_v2 import DeepseekVL2, load_pil_images
# pip install xformers==0.0.28 torchvision==0.19.1 torch==2.4.1 mdtex2html gradio

parser = define_parser()
parser.add_argument("--multi", type=int, default=0)
args = parser.parse_args()

# Load model
t0 = time.time()
model_path = args.model_path
if args.auto_model:
    model = AutoModel.from_pretrained(model_path)
else:
    model = DeepseekVL2(model_path, parallel=True)
    model.load_model_safetensors(model_path)
print(f">>>Load model '{model_path}' finished in {time.time() - t0:.2f} seconds<<<")

# images need download from https://github.com/deepseek-ai/DeepSeek-VL2/tree/main/images
if args.multi == 0:
    # example from https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/inference.py
    messages = [
        {
            "role": "<|User|>",
            "content": "<image>\n<image>\n<|grounding|>In the first image, an object within the red rectangle is marked. Locate the object of the same category in the second image.",
            "images": [
                "images/incontext_visual_grounding_1.jpeg",
                "images/icl_vg_2.jpeg"
            ],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
else:
    messages = [
        {
            "role": "<|User|>",
            "content": "This is image_1: <image>\n"
                       "This is image_2: <image>\n"
                       "This is image_3: <image>\n"
                       " Can you tell me what are in the images?",
            "images": [
                "images/multi_image_1.jpeg",
                "images/multi_image_2.jpeg",
                "images/multi_image_3.jpeg",
            ],
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

# messages = [
#     {
#         "role": "<|User|>", "content": "Who are you?"
#     },
#     {"role": "<|Assistant|>", "content": ""}
# ]

config: DynamicBatchConfig = generator_config_from_cmd(args)
gen_arg: GeneratorArg = generator_arg_from_cmd(args)


def save_answer_image(messages, answer):
    if '<|ref|>' in answer and '<|det|>' in answer:
        print(f"Try save answer image...")
        pil_images = load_pil_images(messages)

        cwd = os.getcwd()
        import deepseek_vl2
        os.chdir(os.path.dirname(deepseek_vl2.__file__) + "/..")  # To found font for parse_ref_bbox
        from deepseek_vl2.serve.app_modules.utils import parse_ref_bbox
        vg_image = parse_ref_bbox(answer, image=pil_images[-1])
        os.chdir(cwd)

        vg_image.save("./vg.jpg", format="JPEG", quality=85)


with DynamicBatchGenerator(config, model) as generator:
    for _ in range(args.round):
        req_result = generator.generate(messages, gen_arg)
        print(req_result)
        save_answer_image(messages, req_result.outputs[0].text)
