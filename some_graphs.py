import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('ml_plt1.pkl', 'rb') as f:
    ml_plt1 = pickle.load(f)
    print(f"hola {ml_plt1[0][:]}")
    for i, row in enumerate(ml_plt1):

        print(f"Row {i} val: {row}")

    # plt.scatter(y = ml_plt1[0][:], x= list(range(len(ml_plt1[0][:]))),label = "row0")
    # plt.scatter(y=ml_plt1[1][:], x=list(range(len(ml_plt1[1][:]))), label="row1")
    # plt.scatter(y=ml_plt1[2][:], x=list(range(len(ml_plt1[2][:]))), label="row2")
    # plt.scatter(y=ml_plt1[3][:], x=list(range(len(ml_plt1[3][:]))), label="row3")
    # plt.plot(ml_plt1[1][:], label="row1",  alpha=0.7)
    # plt.plot(ml_plt1[1][:], label="row1", alpha=0.7)
    # plt.plot(ml_plt1[2][:], label="row0", alpha=0.5)
    # plt.plot(ml_plt1[3][:], label="row0", alpha=1)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # 4 rows, 1 column
    # ["slow.csv", "fast.csv", "typical.csv", "treated.csv"]
    axs[0].hist(ml_plt1[0][:], label='Slow corner', color='navy', bins=100, edgecolor='black')
    axs[1].hist(ml_plt1[1][:], label='Fast Corner', color='blue', bins=100, edgecolor='black')
    axs[2].hist(ml_plt1[2][:], label='Typical Corner', color='slateblue', bins=100, edgecolor='black')
    axs[3].hist(ml_plt1[3][:], label='All', color='mediumpurple', bins=100, edgecolor='black')


    # Add labels, titles and legends for each subplot
    for i, ax in enumerate(axs):
        #ax.set_ylabel(f'Row {i} Values')
        ax.set_ylim(0,80)
        ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("MSE OpenLane histogram", fontsize=16)

    #plt.show()

    with open('opl_plt2.pkl', 'rb') as f:
        opl_plt2 = pickle.load(f)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # 4 rows, 1 column
        # ["slow.csv", "fast.csv", "typical.csv", "treated.csv"]
        axs[0].hist(opl_plt2[0][:], label='Slow corner ML', color='red', bins=100, edgecolor='black')
        axs[1].hist(opl_plt2[1][:], label='Fast Corner ML', color='darkred', bins=100, edgecolor='black')
        axs[2].hist(opl_plt2[2][:], label='Typical Corner ML', color='firebrick', bins=100, edgecolor='black')
        axs[3].hist(opl_plt2[3][:], label='All ML', color='lightcoral', bins=100, edgecolor='black')
        # axs[0].plot(ml_plt1[0][:], label='Slow corner OPL', color='navy')
        # axs[1].plot(ml_plt1[1][:], label='Fast Corner OPL', color='blue')
        # axs[2].plot(ml_plt1[2][:], label='Typical Corner OPL', color='slateblue')
        # axs[3].plot(ml_plt1[3][:], label='All OPL', color='mediumpurple')

        # Add labels, titles and legends for each subplot
        for i, ax in enumerate(axs):
            # ax.set_ylabel(f'Row {i} Values')
            ax.set_ylim(0, 80)
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.suptitle("MSE Model histogram", fontsize=16)

        #plt.show()

with open('plt4.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)
    odd_indices = [data[i] for i in range(len(data)) if i % 2 == 1]
    even_indices = [data[i] for i in range(len(data)) if i % 2 == 0]

    # Convert to numpy arrays for easier manipulation
    odd_indices = np.array(odd_indices)
    even_indices = np.array(even_indices)
    #
    # # Create the bar width
    # bar_width = 0.35
    # index = np.arange(len(even_indices))  # Position of bars on x-axis
    #
    # # Plotting
    # fig, ax = plt.subplots()
    #
    # bars1 = ax.bar(index - bar_width / 2, even_indices[:, 0], bar_width, label='Even Index - Category 1', color='b')
    # bars2 = ax.bar(index + bar_width / 2, even_indices[:, 1], bar_width, label='Even Index - Category 2', color='c')
    # bars3 = ax.bar(index + bar_width / 2 + bar_width, odd_indices[:, 0], bar_width, label='Odd Index - Category 1',
    #                color='r')
    # bars4 = ax.bar(index + bar_width / 2 + bar_width * 2, odd_indices[:, 1], bar_width, label='Odd Index - Category 2',
    #                color='orange')
    #
    # # Add labels and title
    # ax.set_xlabel('Groups')
    # ax.set_ylabel('Values')
    # ax.set_title('Bar Graph of Even and Odd Indexed Data')
    # ax.set_xticks(index + bar_width)
    # ax.set_xticklabels([f'Group {i + 1}' for i in range(len(index))])
    # ax.legend()
    #
    # plt.show()
    x_labels = ["Slow Corner", "Typical Corner", "Fast Corner", 'All']

    # data from https://allisonhorst.github.io/palmerpenguins/

    import matplotlib.pyplot as plt
    import numpy as np

    species = ("Model", "OpenLane")
    penguin_means = {
        'Slow Corner': (1.18, 1.55),
        'Fast Corner': (1.37, 1.66),
        'Typical Corner': (0.55, 1.47),
        'All Corners': (1.32, 1.87),
    }

    x = np.arange(len(species))  # the label locations
    width = 0.175  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Length (mm)')
    ax.set_title('Average RMSE for ML and OPL')
    ax.set_xticks(x + width, species)
    ax.legend(loc='upper left', ncols=3)
    ax.set_ylim(0, 2.5)

    plt.show()