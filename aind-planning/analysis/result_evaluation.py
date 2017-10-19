import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def barplot(problem, title):
    df = pd.DataFrame(problem)
    indices = np.arange(len(df))
    width = 0.2

    fig, ax = plt.subplots()
    ax.barh(indices + width * 0.5, df['expansions'], width, label='#Expansions')
    ax.barh(indices + width * 1.5, df['goal_tests'], width, label='#Goal Tests')
    ax.barh(indices + width * 2.5, df['new_nodes'], width, label='#New Nodes')
    ax.barh(indices + width * 3.5, df['plan_length'], width, label='Plan Length')

    ax.set(yticks=indices + width, yticklabels=df['graph'], ylim=[4*width - 1, len(df)])
    ax.legend()
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    problem_1 = {
        'graph': ['BFS', 'DFGS', 'UCS', 'ASTAR1', 'ASTAR2'],
        'expansions': [43, 21, 55, 41, 11],
        'goal_tests': [56, 22, 57, 43, 13],
        'new_nodes': [180, 84, 224, 170, 50],
        'plan_length': [6, 20, 6, 6, 6]
    }
    barplot(problem_1, 'Problem 1')

    problem_2 = {
        'graph': ['BFS', 'DFGS', 'UCS', 'ASTAR1', 'ASTAR2'],
        'expansions': [3346, 107, 4853, 1450, 86],
        'goal_tests': [4612, 108, 4855, 1452, 88],
        'new_nodes': [30534, 959, 44041, 13303, 841],
        'plan_length': [9, 105, 9, 9, 9]
    }
    barplot(problem_2, 'Problem 2')

    problem_3 = {
        'graph': ['BFS', 'DFGS', 'UCS', 'ASTAR1', 'ASTAR2'],
        'expansions': [14120, 292, 18223, 5040, 315],
        'goal_tests': [17673, 293, 18225, 5042, 317],
        'new_nodes': [124926, 2388, 159618, 44944, 2902],
        'plan_length': [12, 288, 12, 12, 12]
    }
    barplot(problem_3, 'Problem 3')
