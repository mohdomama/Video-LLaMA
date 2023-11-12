from __future__ import annotations

import warnings

import clip
import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
import itertools
import tqdm
import pickle
from prepositions import preposition_a, preposition_a_and_b, preposition_a_and_b_and_c, preposition_a_until_b

warnings.filterwarnings("ignore")


class ClipPerception():
    """Yolo."""

    def __init__(self, config: DictConfig, weight_path: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self._config = config
        # self._classes_reversed = {v: k for k, v in self.model.names.items()}

    def load_model(self, weight_path) -> None:
        """Not Needed For CLIP."""
        pass

    def _parse_class_name(self, class_names: list[str]) -> list[str]:
        """Parse class name.

        Args:
            class_names (list[str]): List of class names.

        Returns:
            list[str]: List of class names.
        """
        return [f"all {class_name}s" for class_name in class_names]

    def detect(self, frame_img: np.ndarray, classes: list) -> any:
        """Detect object in frame.

        Args:
            frame_img (np.ndarray): Frame image.
            classes (list[str]): List of class names.

        Returns:
            any: Detections.
        """
        image = Image.fromarray(frame_img.astype("uint8"), "RGB")
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = clip.tokenize(classes).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            scores = image_features @ text_features.t()
            scores = scores[0].detach().cpu().numpy()

            # logits_per_image, logits_per_text = self.model(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        self._detection = None  # Todo: Figure out what to do about it

        self._confidence = scores

        self._size = len(scores)

        return self._detection

    def get_confidence_score(self, frame_img: np.ndarray, true_label: str) -> any:
        # TODO: What is this about? It was not being called for YOLO
        self.detect(frame_img, true_label)
        return float(self._confidence[0])
    
    def get_confidence_score_batch(self, frame: np.ndarray, true_label: str) -> any:
        # TODO: What is this about? It was not being called for YOLO
        images = []
        for frame_img in frame:
            image = Image.fromarray(frame_img.astype("uint8"), "RGB")
            image = self.preprocess(image).to(self.device)
            images.append(image)
        images = torch.stack(images).to(self.device)
        text = clip.tokenize(true_label).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)

            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            scores = image_features @ text_features.t()
            scores = scores.detach().cpu().numpy()
        
        return scores


preposition_to_func = {
        'A': preposition_a,
        'AandB': preposition_a_and_b,
        'AandBandC': preposition_a_and_b_and_c,
        'AuntilB': preposition_a_until_b
    }

detection_array = {
        'A': np.array([-1]),
        'AandB': np.array([-1, -1]),
        'AandBandC': np.array([-1, -1, -1]),
        'AuntilB': np.array([-1, -1])
    }

number_of_prepos = {
        'A': 1,
        'AandB': 2,
        'AandBandC': 3,
        'AuntilB': 2
    }


def run_and_test():
    model = ClipPerception(None, None)
    possible_lengths = [5,10,15,20, 25]
    repeat = 10
    img_names_perm  = list(itertools.permutations(['horse', 'aeroplane', 'dog', 'cat']))
    num_perm = len(img_names_perm)
    data_path = '/home/mo25464/work/Video-LLaMA/data/'

    device = torch.device('cuda')
    conf_thresh  = 0.25
    
    # for preposition_to_use in ['A', 'AandB', 'AandBandC']:
    for preposition_to_use in ['A', 'AandB', 'AandBandC']:
        print(f'Running {preposition_to_use}')
        results = np.zeros(( num_perm, len(possible_lengths), repeat ))
        for perm_idx, img_names in enumerate(tqdm.tqdm(img_names_perm)):
            for video_len in possible_lengths:
                for repeat_itr in range(repeat):
                    final_tensor, ground_truth, msg, question, final_list , IV_texts = preposition_to_func[preposition_to_use](video_len=video_len, img_names=img_names, InternVideo=True, data_path=data_path)
                    frames = [np.array(x) for x in final_list]

                    scores = model.get_confidence_score_batch(frames, img_names[:number_of_prepos[preposition_to_use]])
                    det_array = detection_array[preposition_to_use].copy().astype(np.float32)
                    for propnum in range(number_of_prepos[preposition_to_use]):
                        for frame_idx, frame in enumerate(frames):
                            conf = scores[frame_idx][propnum]
                            # if conf > conf_thresh:
                            #     det_array[propnum] = 1     
                            det_array[propnum] = conf

                        if not np.any(det_array == -1):        
                            results[perm_idx, possible_lengths.index(video_len), repeat_itr] = np.mean(det_array)


                    

        with open(f'{data_path}_nsvstl_plotdata_{preposition_to_use}.pkl', 'wb') as f:
            pickle.dump([results, possible_lengths, repeat], f)

def run_until_test():
    model = ClipPerception(None, None)
    possible_lengths = [5,10,15,20, 25]
    repeat = 10
    img_names_perm  = list(itertools.permutations(['horse', 'aeroplane', 'dog', 'cat']))
    num_perm = len(img_names_perm)
    data_path = '/home/mo25464/work/Video-LLaMA/data/'

    device = torch.device('cuda')
    conf_thresh  = 0.25
    
    # for preposition_to_use in ['A', 'AandB', 'AandBandC']:
    for preposition_to_use in ['AuntilB']:
        print(f'Running {preposition_to_use}')
        results = np.zeros(( num_perm, len(possible_lengths), repeat ))
        for perm_idx, img_names in enumerate(tqdm.tqdm(img_names_perm)):
            for video_len in possible_lengths:
                for repeat_itr in range(repeat):
                    final_tensor, ground_truth, msg, question, final_list , IV_texts, answer = preposition_to_func[preposition_to_use](video_len=video_len, img_names=img_names, InternVideo=True, data_path=data_path)
                    frames = [np.array(x) for x in final_list]

                    scores = model.get_confidence_score_batch(frames, img_names[:number_of_prepos[preposition_to_use]])
                    det_array = detection_array[preposition_to_use].copy().astype(np.float32)
                    for propnum in range(number_of_prepos[preposition_to_use]):
                        for frame_idx, frame in enumerate(frames):
                            conf = scores[frame_idx][propnum]
                            # if conf > conf_thresh:
                            #     det_array[propnum] = 1     
                            det_array[propnum] = conf

                        if not np.any(det_array == -1):        
                            results[perm_idx, possible_lengths.index(video_len), repeat_itr] = np.mean(det_array)


                    

        with open(f'{data_path}_nsvstl_plotdata_{preposition_to_use}.pkl', 'wb') as f:
            pickle.dump([results, possible_lengths, repeat], f)

def main():
    test = ClipPerception(None, None)
    img = cv2.imread("/opt/Neuro-Symbolic-Video-Frame-Search/store/omama/cat1.jpeg")
    test.detect(img, ["cat", "dog", "person", "table"])
    print("Testing with 4 labels:")
    print(test._confidence)

    print("\nTesting with 1 label:")
    test.detect(img, ["cat"])
    print(test._confidence)
    breakpoint()


if __name__ == "__main__":
    # run_and_test()
    run_until_test()
