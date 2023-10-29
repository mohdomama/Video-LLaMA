from matplotlib import pyplot as plt
import numpy as np
import pickle

preposition = 'AandBandC'

# Load image from pickle file
with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
    results, possible_lengths, possible_ratio = pickle.load(f)


# plot accuracy vs ration
import matplotlib.pyplot as plt
vals = np.average(results, axis=0)    
plt.scatter(possible_ratio, vals, s=55, c='g')
# plot title, x and y labels
plt.title(f"Finding Existence of a Preposition({preposition}) in a Video using VideoLlama")
plt.xlabel("Ratio of Frames Containing the Preposition", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.savefig(f'data/{preposition}_ratio_vs_accuracy.png')

plt.cla()

vals = np.average(results, axis=1)    
plt.scatter(possible_lengths, vals)
# plot title, x and y labels
plt.title("Finding Existence of a Preposition ({main_img}) in a Video using VideoLlama")
plt.xlabel("Length on Video")
plt.ylabel("Accuracy")
plt.savefig(f'data/{preposition}_length_vs_accuracy.png')