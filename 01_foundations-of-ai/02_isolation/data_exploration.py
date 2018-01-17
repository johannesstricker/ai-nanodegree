import isolation
import sample_players
from competition_agent import CustomPlayer, custom_score, remove_equal_moves
import timeit
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import seaborn as sns
import os.path
import keyboard
import sys

def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def score_sided(game, active_player):
    # Get player and opponent location.
    own_location = game.get_player_location(active_player)
    opp_location = game.get_player_location(game.get_opponent(active_player))
    if own_location is None or opp_location is None:
        return 0

    # Calculate the distance between the player's positions and the board's center.
    center = (game.height / 2, game.width / 2)
    value = 0
    # Check if we are both on the same vertical side of center.
    if (own_location[0] < center[0]) == (opp_location[0] < center[0]):
        # Add the difference between the players' distances to the board's center on the vertical axis.
        value += abs(center[0] - opp_location[0]) - abs(center[0] - own_location[0])
    # Check if we are both on the same horizontal side of center.
    if (own_location[1] < center[1]) == (opp_location[1] < center[1]):
        # Add the difference between the players' distances to the board's center on the horizontal axis.
        value += abs(center[1] - opp_location[1]) - abs(center[1] - own_location[1])
    return value

def score_delta_moves(game, active_player):
    # Return the difference between the number of available moves.
    # The opponents moves are scaled by two, to punish him more when he
    # has zero moves left.
    own_moves = len(game.get_legal_moves(active_player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(active_player)))
    delta_moves = own_moves - 2 * opp_moves;
    return delta_moves

def score_delta_moves_cubed(game, active_player):
    return score_delta_moves(game, active_player) ** 3

def score_center_distance(game, active_player):
    # Get player and opponent location.
    own_location = game.get_player_location(active_player)
    opp_location = game.get_player_location(game.get_opponent(active_player))
    if own_location is None or opp_location is None:
        return 0
    # Calculate the distance between the player's positions and the board's center.
    center = (game.height / 2, game.width / 2)
    own_center_dist = distance(own_location, center)
    opp_center_dist = distance(opp_location, center)
    return opp_center_dist - own_center_dist

def score_center_distance_cubed(game, active_player):
    return score_center_distance(game, active_player) ** 3

def score_opponent_blocked(game, active_player):
    center = (game.height / 2, game.width / 2)
    player = game.get_player_location(active_player)
    opponent = game.get_player_location(game.get_opponent(active_player))
    # Ensure that both players have already made their first move.
    if player is None or opponent is None:
        return 0
    if player[0] == opponent[0] and abs(player[1] - opponent[1]) == 1:
        return 1
    if player[1] == opponent[1] and abs(player[0] - opponent[0]) == 1:
        return 1
    return 0

def displayWinningCorrelation(df):
    # corr = df.drop(labels='win_A', axis=1).corrwith(samples['win_A'])
    # corr = pd.DataFrame(corr, columns=['win_A'])
    corr = df.corr()
    print(corr)
    ax = sns.heatmap(corr, annot=True, linewidths=.5, square=True)
    sns.plt.xticks(rotation=90)
    sns.plt.yticks(rotation=0)
    sns.plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.3)
    sns.plt.show()

def recordGameSamples(game_count = 1000):
    player = sample_players.RandomPlayer()
    opponent = sample_players.RandomPlayer()
    samples = { 'Custom1': [], 'Custom1³': [], 'Custom2': [], 'Custom2³': [], 'Custom3': [], 'Block Opponent': [], 'Win%': [] };

    for i in range(0, game_count):
        # Initialize a new game.
        game = isolation.Board(player, opponent)
        game_depth = 1
        closer_to_center = 0
        closer_to_center_cubed = 0
        opponent_sided = 0
        opponent_blocked = 0
        delta_moves = 0
        delta_moves_cubed = 0

        # Play a game with random moves and record the data.
        while True:
            legal_player_moves = game.get_legal_moves()

            # Get the current player's next move.
            curr_move = game._active_player.get_move(game, lambda x: 10000)
            if curr_move is None:
                curr_move = Board.NOT_MOVED

            # If the game is over, store the winner.
            if curr_move not in legal_player_moves:
                samples['Custom2'].append(closer_to_center)
                samples['Custom2³'].append(closer_to_center_cubed)
                samples['Custom3'].append(opponent_sided)
                samples['Block Opponent'].append(opponent_blocked)
                samples['Custom1'].append(delta_moves)
                samples['Custom1³'].append(delta_moves_cubed)
                samples['Win%'].append(1 if game.is_winner(player) else 0)
                break;

            game.apply_move(curr_move)
            # Record features after player's move.
            if game._inactive_player == player and game_depth >= 2:
                closer_to_center += score_center_distance(game, player)
                closer_to_center_cubed += score_center_distance_cubed(game, player)
                opponent_sided += score_sided(game, player)
                opponent_blocked += score_opponent_blocked(game, player)
                delta_moves += score_delta_moves(game, player)
                delta_moves_cubed += score_delta_moves_cubed(game, player)
            game_depth += 1

        # Output progress.
        print('\r{}%'.format(i*100//game_count), end='')
        sys.stdout.flush()
    return pd.DataFrame.from_dict(samples)


if __name__ == '__main__':
    if os.path.isfile('samples.csv'):
        df = pd.DataFrame.from_csv('samples.csv')
    else:
        df = recordGameSamples(10000)
        df.to_csv('samples.csv')
    # df['custom_1'] = df.apply(score_custom, axis=1)
    # df['custom_2'] = df.apply(lambda row: 2 * row['delta_moves'] + row['opponent_sided'], axis=1)
    # df['custom_3'] = df.apply(lambda row: 2 * row['delta_moves'] + row['opponent_sided'] + 2 * row['opponent_blocked'], axis=1)
    print(df.describe())
    displayWinningCorrelation(df)
