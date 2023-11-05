import pickle
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def plot_combined():
    prepositions = ['A', 'AandB', 'AandBandC']
    colors = ['r', 'g', 'b']

    for idx, preposition in enumerate(prepositions):
        # Load image from pickle file
        with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=1)    
        plt.plot(possible_lengths, vals, colors[idx])
        plt.xlabel("Length of Video", fontsize=13)
        plt.ylabel("Accuracy", fontsize=13)

    # set legends
    plt.title(f"Finding Existence of a Preposition in a Video using VideoLlama")
    plt.legend(prepositions, loc='lower left')
    plt.savefig(f'data/Combined_length_vs_accuracy.png')


def plot_combined_with_variance():
    prepositions = ['A', 'AandB', 'AandBandC', 'AuntilB']
    colors = ['r', 'g', 'b', 'orange']

    for idx, preposition in enumerate(prepositions):
        # Load image from pickle file
        with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
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
    plt.title(f"Finding Existence of a Preposition in a Video using VideoLlama")
    plt.legend(prepositions, loc='lower left')
    plt.savefig(f'data/Combined_length_vs_accuracy.png')

def plot_until():
    prepositions = ['AuntilB']
    colors = ['r', 'g', 'b']

    for idx, preposition in enumerate(prepositions):
        # Load image from pickle file
        with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
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
    plt.title(f"Finding Existence of a Preposition in a Video using VideoLlama")
    plt.legend(prepositions, loc='lower left')
    plt.savefig(f'data/Until_length_vs_accuracy.png')



def plot_combined_with_boxplot():
    prepositions = ['A', 'AandB', 'AandBandC']
    colors = ['r', 'g', 'b']

    data = []

    for idx, preposition in enumerate(prepositions):
        # Load image from pickle file
        with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)
        data.append(vals)

    # Plot the box plot
    plt.boxplot(data, labels=prepositions, patch_artist=True)

    # Add labels and title to the plot
    plt.xlabel("Preposition", fontsize=13)
    plt.ylabel("Accuracy", fontsize=13)
    plt.title(f"Finding Existence of a Preposition in a Video using VideoLlama")

    # Set colors for the box plot
    for patch, color in zip(plt.gca().artists, colors):
        patch.set_facecolor(color)

    # Save and show the plot
    plt.savefig(f'data/Combined_preposition_vs_accuracy.png')
    plt.show()


def plot_seaborn():
    prepositions = ['A', 'AandB', 'AandBandC']
    colors = ['r', 'g', 'b']

    data = []
    for idx, preposition in enumerate(prepositions):
        # Load image from pickle file
        with open(f'data/plotdata_{preposition}.pkl', 'rb') as f:
            results, possible_lengths, repeat = pickle.load(f)

        vals = np.average(results, axis=2)
        data.append(vals)

    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    data = np.array(data)
    df1 = pd.DataFrame(data[:, :, 0].T, columns=list(range(1,4))).assign(Trial=1)
    df2 = pd.DataFrame(data[:, :, 1].T, columns=list(range(1,4))).assign(Trial=2)
    df3 = pd.DataFrame(data[:, :, 2].T, columns=list(range(1,4))).assign(Trial=3)
    df4 = pd.DataFrame(data[:, :, 3].T, columns=list(range(1,4))).assign(Trial=4)
    df5 = pd.DataFrame(data[:, :, 4].T, columns=list(range(1,4))).assign(Trial=5)

    cdf = pd.concat([df1, df2, df3, df4, df5])                                # CONCATENATE
    mdf = pd.melt(cdf, id_vars=['Trial'], var_name=['Number'])      # MELT

    print(mdf.head())
    ax = sns.boxplot(x="Trial", y="value", hue="Number", data=mdf)  # RUN PLOT   
    plt.savefig(f'data/Combined_preposition_vs_accuracy.png')
    plt.close()


if __name__ == "__main__":
    # plot_combined()
    plot_combined_with_variance()
    # plot_until()
    # plot_combined_with_boxplot()
    # plot_seaborn()