import matplotlib.pyplot as plt
import numpy as np

categories = ['Star', 'Score', '3-2-1']
voter_behaviors = ['Honest', 'One-sided', 'Strategic']
sets_of_data = {
    'Star': {
        'Honest': [0.4862, 0.6585, 0.5177],
        'One-sided': [1.9073, 1.9864, 2.0358],
        'Strategic': [1.2611, 1.3372, 1.2363]
    },
    'Score': {
        'Honest': [0.4862, 0.6585, 0.5177],
        'One-sided': [3.7147, 3.9140, 3.4996],
        'Strategic': [0.7996, 0.8021, 0.6259]
    },
    '3-2-1': {
        'Honest': [2.5180, 2.0924, 2.9139],
        'One-sided': [3.6335, 3.8423, 3.4622],
        'Strategic': [3.9538, 4.4685, 4.1903]
    }
}

colors = {'Honest': 'blue', 'One-sided': 'green', 'Strategic': 'red'}
markers = ['o', 'o', 'o']

plt.figure(figsize=(12, 8))

for i, category in enumerate(categories):
    for j, behavior in enumerate(voter_behaviors):
        for k, (data_point, marker) in enumerate(zip(sets_of_data[category][behavior], markers)):
            plt.scatter(data_point, category, color=colors[behavior], marker=marker, label=f'{behavior} - Set {k+1}' if i == 0 else "")

plt.xlabel('Bayesian Regret')
plt.ylabel('Voting Method')
plt.title('Bayesian Regret by Voting Method and Voter Behavior')
plt.legend(title='Voter Behavior and Data Set')

plt.show()
