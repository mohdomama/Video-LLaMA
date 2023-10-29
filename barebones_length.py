"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random
from torchvision import transforms
from PIL import Image

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *
import itertools
import tqdm
import pickle



#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# take cat.jpeg and dog.jpeg in data folder and create a pytorch tensor of 8 images with 50% cat and 50% dog at random indices
def create_video_one_dog(video_len: int = 8):
    cat_img = Image.open('data/cat.jpeg')
    dog_img = Image.open('data/horse.jpeg')

    # True means cat
    ground_truth = [True] * (video_len)
    ground_truth[0] = False
    # ground_truth = [False, False] * (video_len//2) # Test only with dog
    # shuffle the list
    random.shuffle(ground_truth)

    final_list = [cat_img.copy() if i else dog_img.copy() for i in ground_truth]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    final_tensor = torch.stack([transform(img) for img in final_list])
    final_tensor = final_tensor.permute(1, 0, 2, 3) * 255
    sec = ", ".join([str(x.item()) for x in torch.arange(video_len) ])
    msg = f"The video contains {video_len} frames sampled at {sec} seconds. Each frame has either a cat or a dog"
    return final_tensor, ground_truth, msg


def create_video_given_split(video_len: int = 8, ratio: float = 0.5):
    main_img = 'horse'
    other_img = 'aeroplane'
    # if video_len%2 != 0: raise ValueError("len must be even!")
    img1 = Image.open('data/' + main_img +'.jpeg')
    img2 = Image.open('data/' + other_img + '.jpeg')

    # True means cat
    half = int(video_len  * ratio)
    ground_truth = [True] * half + [False] * (video_len - half)
    # ground_truth = [False, False] * (video_len//2) # Test only with dog
    # shuffle the list
    random.shuffle(ground_truth)

    final_list = [img1.copy() if i else img2.copy() for i in ground_truth]

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    final_tensor = torch.stack([transform(img) for img in final_list])
    final_tensor = final_tensor.permute(1, 0, 2, 3) * 255
    sec = ", ".join([str(x.item()) for x in torch.arange(video_len) ])
    msg = f"The video contains {video_len} frames sampled at {sec} seconds."
    return final_tensor, ground_truth, msg


def preposition_a(video_len, img_names):
    a_img = Image.open('data/' + img_names[0] +'.jpeg')
    b_img = Image.open('data/' + img_names[1] + '.jpeg')

    asplit = 3
    bsplit = video_len - asplit
    indxs = np.arange(video_len)
    np.random.shuffle(indxs)
    aidxs = indxs[:asplit]
    bidxs = indxs[asplit:]

    final_list = [b_img.copy() for i in range(video_len)]

    for idx in aidxs:
        final_list[idx] = a_img.copy()
    
    ground_truth = np.array(['b'] * video_len)
    ground_truth[aidxs] = 'a'

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    final_tensor = torch.stack([transform(img) for img in final_list])
    final_tensor = final_tensor.permute(1, 0, 2, 3) * 255
    sec = ", ".join([str(x.item()) for x in torch.arange(video_len) ])
    msg = f"The video contains {video_len} frames sampled at {sec} seconds."
    question = f"Is there a {img_names[0]} in this video?"
    return final_tensor, ground_truth, msg, question


def preposition_a_and_b(video_len, img_names):
    a_img = Image.open('data/' + img_names[0] +'.jpeg')
    b_img = Image.open('data/' + img_names[1] + '.jpeg')
    c_img = Image.open('data/' + img_names[2] + '.jpeg')
    

    asplit = 2
    bsplit = 1
    csplit = video_len - asplit - bsplit
    indxs = np.arange(video_len)
    np.random.shuffle(indxs)
    aidxs = indxs[:asplit]
    bidxs = indxs[asplit:asplit+bsplit]
    cidxs = indxs[asplit+bsplit:]


    final_list = [c_img.copy() for i in range(video_len)]

    for idx in aidxs:
        final_list[idx] = a_img.copy()
    
    for idx in bidxs:
        final_list[idx] = b_img.copy()

    ground_truth = np.array(['c'] * video_len)
    ground_truth[aidxs] = 'a'
    ground_truth[bidxs] = 'b'

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    final_tensor = torch.stack([transform(img) for img in final_list])
    final_tensor = final_tensor.permute(1, 0, 2, 3) * 255
    sec = ", ".join([str(x.item()) for x in torch.arange(video_len) ])
    msg = f"The video contains {video_len} frames sampled at {sec} seconds."
    question = f"Is there a {img_names[0]} and an {img_names[1]} in the video?"
    return final_tensor, ground_truth, msg, question

def preposition_a_and_b_and_c(video_len, img_names):
    a_img = Image.open('data/' + img_names[0] +'.jpeg')
    b_img = Image.open('data/' + img_names[1] + '.jpeg')
    c_img = Image.open('data/' + img_names[2] + '.jpeg')
    d_img = Image.open('data/' + img_names[3] + '.jpeg')

    asplit = 1
    bsplit = 1
    csplit = 1
    dsplit = video_len - asplit - bsplit - csplit

    indxs = np.arange(video_len)
    np.random.shuffle(indxs)
    aidxs = indxs[:asplit]
    bidxs = indxs[asplit:asplit+bsplit]
    cidxs = indxs[asplit+bsplit:asplit+bsplit+csplit]
    didxs = indxs[asplit+bsplit+csplit:]

    final_list = [d_img.copy() for i in range(video_len)]

    for idx in aidxs:
        final_list[idx] = a_img.copy()
    
    for idx in bidxs:
        final_list[idx] = b_img.copy()
    
    for idx in cidxs:
        final_list[idx] = c_img.copy()

    ground_truth = np.array(['d'] * video_len)
    ground_truth[aidxs] = 'a'
    ground_truth[bidxs] = 'b'
    ground_truth[cidxs] = 'c'

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    final_tensor = torch.stack([transform(img) for img in final_list])
    final_tensor = final_tensor.permute(1, 0, 2, 3) * 255
    sec = ", ".join([str(x.item()) for x in torch.arange(video_len) ])
    msg = f"The video contains {video_len} frames sampled at {sec} seconds."
    question = f"Is there a {img_names[0]}, an {img_names[1]} and a {img_names[2]} in the video? "
    return final_tensor, ground_truth, msg, question

    



# final_tensor, ground_truth, msg = create_video_equal_split(video_len=8)

# ========================================
#             Model Initialization
# ========================================


print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

# possible_lengths = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# possible_ratio = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

possible_lengths = [5,10,15,20, 25]
repeat = 20
img_names_perm  = list(itertools.permutations(['horse', 'aeroplane', 'dog', 'cat']))
num_perm = len(img_names_perm)



preposition_to_use = 'A'
preposition_to_func = {
    'A': preposition_a,
    'AandB': preposition_a_and_b,
    'AandBandC': preposition_a_and_b_and_c,
}

for preposition_to_use in ['A', 'AandB', 'AandBandC']:
    results = np.zeros(( num_perm, len(possible_lengths), repeat ))
    for perm_idx, img_names in enumerate(tqdm.tqdm(img_names_perm)):
        for video_len in possible_lengths:
            for repeat_itr in range(repeat):
                chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
                chat_state = conv_llava_llama_2.copy()

                chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
                img_list = []

                final_tensor, ground_truth, msg, question = preposition_to_func[preposition_to_use](video_len=video_len, img_names=img_names)

                llm_message = chat.custom_upload_video_without_audio(final_tensor, msg, chat_state, img_list)

                user_message = question

                chat.ask(user_message, chat_state)
                llm_message = chat.answer(conv=chat_state,
                                        img_list=img_list,
                                        num_beams=5, # Set with slider in demo
                                        temperature=1, # Set with slider in demo
                                        max_new_tokens=300,
                                        max_length=2000)[0]
                # print('LLM Message: ', llm_message)
                if 'Yes' in llm_message and 'No' not in llm_message and 'no' not in llm_message:
                    results[perm_idx, possible_lengths.index(video_len), repeat_itr] = 1


    # print('Results:', results)
    with open(f'data/plotdata_{preposition_to_use}.pkl', 'wb') as f:
        pickle.dump([results, possible_lengths, repeat], f)