import numpy as np
from typing import List, Tuple
from collections import defaultdict

class Voter:
    def __init__(self, position: np.ndarray):
        self.position = position

class Candidate:
    def __init__(self, position: np.ndarray):
        self.position = position

def generate_voters_and_candidates(num_voters: int, num_candidates: int, num_issues: int) -> Tuple[List[Voter], List[Candidate]]:
    voters = [Voter(np.random.normal(0, 1, num_issues)) for _ in range(num_voters)]
    candidates = [Candidate(np.random.normal(0, 1, num_issues)) for _ in range(num_candidates)]
    return voters, candidates

def utility(voter: Voter, candidate: Candidate) -> float:
    return -np.linalg.norm(voter.position - candidate.position)

def zero_to_five_star_honest(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    min_util, max_util = min(utilities), max(utilities)
    return [round(5 * (u - min_util) / (max_util - min_util)) if max_util != min_util else 5 for u in utilities]

def zero_to_five_star_one_sided(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    favorite = np.argmax(utilities)
    second_favorite = np.argsort(utilities)[-2]
    return [5 if i == favorite else (1 if i == second_favorite else 0) for i in range(len(candidates))]

def zero_to_five_star_strategic(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    sorted_indices = np.argsort(utilities)[::-1]
    votes = [0] * len(candidates)
    votes[sorted_indices[0]] = 5
    votes[sorted_indices[-1]] = 0
    if len(candidates) > 2:
        for i in range(1, len(candidates) - 1):
            votes[sorted_indices[i]] = 3
    return votes

def zero_to_five_score_honest(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    min_util, max_util = min(utilities), max(utilities)
    if max_util == min_util:
        return [5] * len(candidates)
    return [round(5 * (u - min_util) / (max_util - min_util)) for u in utilities]

def zero_to_five_score_one_sided(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    favorite = np.argmax(utilities)
    return [5 if i == favorite else 0 for i in range(len(candidates))]

def zero_to_five_score_strategic(voter: Voter, candidates: List[Candidate]) -> List[int]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    sorted_indices = np.argsort(utilities)[::-1]
    votes = [0] * len(candidates)
    votes[sorted_indices[0]] = 5
    if len(candidates) > 1:
        votes[sorted_indices[1]] = 4
    return votes

def three_two_one_honest(voter: Voter, candidates: List[Candidate]) -> List[str]:

    utilities = [utility(voter, candidate) for candidate in candidates]

    votes = []
    for u in utilities:
        if 0 >= u > -1.5:
            votes.append('Good')
        elif -1.5 >= u > -3:
            votes.append('OK')
        else:
            votes.append('Bad')

    return votes


def three_two_one_one_sided(voter: Voter, candidates: List[Candidate]) -> List[str]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    favorite = np.argmax(utilities)
    votes = ['Bad'] * len(candidates)
    votes[favorite] = 'Good'
    return votes

def three_two_one_strategic(voter: Voter, candidates: List[Candidate]) -> List[str]:
    utilities = [utility(voter, candidate) for candidate in candidates]
    sorted_indices = np.argsort(utilities)[::-1]
    votes = ['Bad'] * len(candidates)
    votes[sorted_indices[0]] = 'Good'

    # Strategic consideration: If there's a clear second choice, mark it as 'OK'
    # Otherwise, mark the least favorite as 'Bad' to increase chances of favorite winning
    if len(candidates) > 2 and utilities[sorted_indices[1]] - utilities[sorted_indices[2]] > 0.1:
        votes[sorted_indices[1]] = 'OK'
    else:
        votes[sorted_indices[-1]] = 'Bad'

    return votes

def run_election(voters: List[Voter], candidates: List[Candidate], voting_method) -> int:
    if voting_method in [three_two_one_honest, three_two_one_one_sided, three_two_one_strategic]:
        return run_three_two_one_election(voters, candidates, voting_method)
    else:
        votes = [voting_method(voter, candidates) for voter in voters]
        total_votes = np.sum(votes, axis=0)
        return np.argmax(total_votes)

def run_three_two_one_election(voters: List[Voter], candidates: List[Candidate], voting_method) -> int:
    votes = [voting_method(voter, candidates) for voter in voters]

    # Count "Good" ratings
    good_counts = defaultdict(int)
    for ballot in votes:
        for i, vote in enumerate(ballot):
            if vote == 'Good':
                good_counts[i] += 1

    # Find 3 semifinalists
    semifinalists = sorted(good_counts, key=good_counts.get, reverse=True)[:3]

    # Count "Bad" ratings for semifinalists, initializing counts to 0
    bad_counts = defaultdict(int) # Use defaultdict to automatically handle missing keys
    for ballot in votes:
        for i in semifinalists:
            if ballot[i] == 'Bad':
                bad_counts[i] += 1

    # Find 2 finalists, handling potential None values in bad_counts
    finalists = sorted(semifinalists, key=lambda i: bad_counts.get(i, 0))[:2] # Provide default value of 0 for missing keys

    # Virtual runoff between finalists
    runoff_counts = {finalist: 0 for finalist in finalists}
    for ballot in votes:
        if ballot[finalists[0]] != ballot[finalists[1]]:
            winner = finalists[0] if ballot[finalists[0]] > ballot[finalists[1]] else finalists[1]
            runoff_counts[winner] += 1

    # Determine winner
    winner = max(runoff_counts, key=runoff_counts.get)
    return winner

def calculate_regret(voters: List[Voter], candidates: List[Candidate], winner: int) -> float:
    best_candidate = max(range(len(candidates)), key=lambda i: sum(utility(voter, candidates[i]) for voter in voters))
    best_utility = sum(utility(voter, candidates[best_candidate]) for voter in voters)
    winner_utility = sum(utility(voter, candidates[winner]) for voter in voters)
    return best_utility - winner_utility

def simulate_bayesian_regret(num_voters: int, num_candidates: int, num_issues: int, num_simulations: int) -> dict:
    voting_methods = {
        "0-5 Star (Honest)": zero_to_five_star_honest,
        "0-5 Star (One-sided)": zero_to_five_star_one_sided,
        "0-5 Star (Strategic)": zero_to_five_star_strategic,
        "0-5 Score (Honest)": zero_to_five_score_honest,
        "0-5 Score (One-sided)": zero_to_five_score_one_sided,
        "0-5 Score (Strategic)": zero_to_five_score_strategic,
        "3-2-1 (Honest)": three_two_one_honest,
        "3-2-1 (One-sided)": three_two_one_one_sided,
        "3-2-1 (Strategic)": three_two_one_strategic
    }

    regrets = {method: [] for method in voting_methods}

    for _ in range(num_simulations):
        voters, candidates = generate_voters_and_candidates(num_voters, num_candidates, num_issues)

        for method, voting_function in voting_methods.items():
            winner = run_election(voters, candidates, voting_function)
            regrets[method].append(calculate_regret(voters, candidates, winner))

    return {method: np.mean(regret_list) for method, regret_list in regrets.items()}

# Run the simulation
num_voters = 200
num_candidates = 5
num_issues = 5
num_simulations = 1000

results = simulate_bayesian_regret(num_voters, num_candidates, num_issues, num_simulations)

for method, regret in results.items():
    print(f"Average Bayesian Regret for {method}: {regret:.4f}")
