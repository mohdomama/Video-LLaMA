import numpy as np
import os
from torchvision import transforms
from PIL import Image
import torch

def preposition_a(video_len, img_names, InternVideo=False, data_path='data/'):
    a_img = Image.open(data_path + img_names[0] +'.jpeg')
    b_img = Image.open(data_path + img_names[1] + '.jpeg')

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
    if not InternVideo:
        return final_tensor, ground_truth, msg, question
    else:
        IV_texts = [f'There is {img_names[0]} in the video',
                    f'There is only {img_names[1]} in the video',]

        return final_tensor, ground_truth, msg, question, final_list , IV_texts


def preposition_a_and_b(video_len, img_names, InternVideo=False, data_path='data/'):
    a_img = Image.open(data_path + img_names[0] +'.jpeg')
    b_img = Image.open(data_path + img_names[1] + '.jpeg')
    c_img = Image.open(data_path + img_names[2] + '.jpeg')
    

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

    if not InternVideo:
        return final_tensor, ground_truth, msg, question
    else:
        IV_texts = [f'There is {img_names[0]} and {img_names[1]} in the video',
                    f'There is only {img_names[2]} in the video',]
        return final_tensor, ground_truth, msg, question, final_list, IV_texts

def preposition_a_and_b_and_c(video_len, img_names, InternVideo=False, data_path='data/'):
    a_img = Image.open(data_path + img_names[0] +'.jpeg')
    b_img = Image.open(data_path + img_names[1] + '.jpeg')
    c_img = Image.open(data_path + img_names[2] + '.jpeg')
    d_img = Image.open(data_path + img_names[3] + '.jpeg')

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

    if not InternVideo:
        return final_tensor, ground_truth, msg, question
    else:
        IV_texts = [f'There is {img_names[0]}, {img_names[1]} and {img_names[2]} in the video',
                    f'There is only {img_names[3]} in the video',]
        return final_tensor, ground_truth, msg, question, final_list, IV_texts

    

def preposition_a_until_b(video_len, img_names, InternVideo=False, data_path='data/'):
    a_img = Image.open(data_path + img_names[0] +'.jpeg')
    b_img = Image.open(data_path + img_names[1] + '.jpeg')
    c_img = Image.open(data_path + 'whitebg' + '.jpeg')
    

    asplit = 1
    bsplit = 1
    csplit = video_len - asplit - bsplit
    indxs = np.arange(video_len)
    # np.random.shuffle(indxs)

    # np select two random indices
    aidxs, bidxs = np.random.randint(0, video_len-1, (2,))

    aidxs = [aidxs]
    bidxs = [bidxs]


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
    question = f"Does this video show a {img_names[0]} until a {img_names[1]} appears?"
    answer = 'yes' if aidxs[0] < bidxs[0] else 'no'
    if not InternVideo:
        return final_tensor, ground_truth, msg, question, answer
    else:
        IV_texts = [f'There is {img_names[0]} and after that {img_names[1]} appears',
                    f'There is {img_names[1]} and after that {img_names[0]} appears',
                    f'There video has only white background',               
                    f'There is {img_names[0]} and {img_names[1]}',               
                    ]
        return final_tensor, ground_truth, msg, question, final_list, IV_texts, answer