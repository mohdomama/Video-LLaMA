import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from swarm_visualizer.boxplot import plot_paired_boxplot
from swarm_visualizer.utility.general_utils import save_fig, set_plot_properties

def plot_combined():
    Propositions = ['A', 'AandB', 'AandBandC']
    colors = ['r', 'g', 'b']

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=1)    
        plt.plot(possible_lengths, vals, colors[idx])
        plt.xlabel("Length of Video", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)

    # set legends
    plt.title(f"Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc='lower left')
    plt.savefig(f'data/Combined_length_vs_accuracy.png')


def plot_combined_with_variance():
    Propositions = ['A', 'AandB', 'AandBandC', 'AuntilB']
    colors = ['r', 'g', 'b', 'orange']

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/_viclip_plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)    
        avg = np.average(vals, axis=0)
        var = np.var(vals, axis=0)
        std = np.std(vals, axis=0)
        upper = avg + std
        lower = avg - std

        # plt.plot(possible_lengths, avg, colors[idx])
        # plt.fill_between(possible_lengths, lower, upper, color=colors[idx], alpha=0.05)
        # plt.plot(possible_lengths, vals, colors[idx])
        plt.errorbar(possible_lengths, avg, yerr=var, fmt=colors[idx])
        plt.xlabel("Length of Video", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)

    # set legends
    plt.title(f"Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc='lower left')
    plt.savefig(f'data/_viclip_Combined_length_vs_accuracy.png')

def plot_until():
    Propositions = ['AuntilB']
    colors = ['r', 'g', 'b']

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        breakpoint()
        vals = np.average(results, axis=2)    
        avg = np.average(vals, axis=0)
        var = np.var(vals, axis=0)
        std = np.std(vals, axis=0)
        upper = avg + std
        lower = avg - std

        # plt.plot(possible_lengths, avg, colors[idx])
        # plt.fill_between(possible_lengths, lower, upper, color=colors[idx], alpha=0.05)
        # plt.plot(possible_lengths, vals, colors[idx])
        plt.errorbar(possible_lengths, avg, yerr=var, fmt=colors[idx])
        plt.xlabel("Length of Video", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)

    # set legends
    plt.title(f"Finding Existence of a Proposition in a Video using VideoLlama")
    plt.legend(Propositions, loc='lower left')
    plt.savefig(f'data/Until_length_vs_accuracy.png')



def plot_combined_with_boxplot():
    Propositions = ['A', 'AandB', 'AandBandC']
    colors = ['r', 'g', 'b']

    data = []

    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)
        data.append(vals)

    # Plot the box plot
    plt.boxplot(data, labels=Propositions, patch_artist=True)

    # Add labels and title to the plot
    plt.xlabel("Proposition", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(f"Finding Existence of a Proposition in a Video using VideoLlama")

    # Set colors for the box plot
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)

    # Save and show the plot
    plt.savefig(f'data/Combined_Proposition_vs_accuracy.png')
    plt.show()


def plot_seaborn():
    set_plot_properties()
    df = None
    Propositions = ['A', 'AandB', 'AandBandC', 'AuntilB']

    # Load VideoLLama data
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        avg = np.average(results, axis=0) # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df)==None:
                    df = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['VideoLlama']})
                else:
                    df_new = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['VideoLlama']})
                    df = pd.concat([df, df_new], ignore_index=True)
    
    # Load ViClip data
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/_viclip_plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        possible_lengths = [5,10,15,20, 25] # Changing possible lenghts (matching 8 to 5)
        avg = np.average(results, axis=0) # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df)==None:
                    df = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['ViCLIP']})
                else:
                    df_new = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['ViCLIP']})
                    df = pd.concat([df, df_new], ignore_index=True)

    # Load NSVS-TL
    for idx, Proposition in enumerate(Propositions):
        # Load image from pickle file
        with open(f'data/_nsvstl_plotdata_{Proposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        # thresh = 0.2142
        # results[results>=thresh] = 1
        # results[results<thresh] = 0

        # # with open(f'data/_nsvstl_plotdata_{Proposition}.pkl', 'wb') as f:
        # #     pickle.dump([results, possible_lengths, repeat], f)

        possible_lengths = [5,10,15,20, 25] # Changing possible lenghts (matching 8 to 5)
        avg = np.average(results, axis=0) # Averaging over permutations
        for i, length in enumerate(possible_lengths):
            for j in range(repeat):
                if type(df)==None:
                    df = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['NSVS-TL']})
                else:
                    df_new = pd.DataFrame({'Proposition': [Proposition], 'Accuracy': [avg[i][j]], 'Length': [length], 'Approach': ['NSVS-TL']})
                    df = pd.concat([df, df_new], ignore_index=True)

    
    
    fig, ax = plt.subplots(figsize=(15, 8))
    plot_paired_boxplot(df,
                        x_var='Length',
                        y_var='Accuracy',
                        hue='Approach',
                        title_str = 'Accuracy of Baselines Drop with Video Length',

                        ax=ax)
    ax.set_xlabel('Length (Frames)')
    plt.legend(fontsize=15)
    plt.savefig(f'data/test_box_accuracy_vs_length.png')
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 8))
    plot_paired_boxplot(df,
                        x_var='Proposition',
                        y_var='Accuracy',
                        hue='Approach',
                        title_str = 'Accuracy of Baselines Drop with Proposition Complexity',
                        ax=ax)
    plt.legend(fontsize=15)
    plt.savefig(f'data/test_box_accuracy_vs_Proposition.png')
    plt.close()

if __name__ == "__main__":
    # plot_combined()
    # plot_combined_with_variance()
    # plot_until()
    # plot_combined_with_boxplot()
    plot_seaborn()