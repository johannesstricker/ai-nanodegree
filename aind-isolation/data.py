import isolation
import sample_players
from competition_agent import CustomPlayer, custom_score, remove_equal_moves
import timeit
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from matplotlib import pyplot as plt

class Timer:
    def __init__(self, time_limit_ms):
        self.timer_start_ms = 1000 * timeit.default_timer()
        self.time_elapsed_ms = lambda: 1000 * timeit.default_timer() - self.timer_start_ms
        self.time_limit_ms = time_limit_ms

    def time_left_ms(self):
        return self.time_limit_ms - self.time_elapsed_ms()

def createOpeningBook(score_fn):
    player = CustomPlayer()
    opponent = sample_players.RandomPlayer()
    game = isolation.Board(opponent, player)

    # Find all possible opponent moves.
    opponent_moves = game.get_legal_moves()
    opponent_moves = remove_equal_moves(game, opponent_moves)
    print('Opponent has {} legal moves.'.format(len(opponent_moves)))

    # Iterate over all possible opponent moves and find the best response.
    opening_book = dict()
    for idx, move in enumerate(opponent_moves):
        forecast = game.forecast_move(move)
        timer = Timer(2000)
        response = player.get_move(forecast, timer.time_left_ms)
        opening_book[move] = response
        print('{}/{} - search depth was {}'.format(idx + 1, len(opponent_moves), player.max_depth_searched))
    print(opening_book)

def plot_corr(df,size=10):
    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot'''

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.show()

def recordGameSamples():
    player = sample_players.RandomPlayer()
    opponent = sample_players.RandomPlayer()

    samples = pd.DataFrame({
        'num_moves_made': [],
        'num_available_moves': [],
        'num_available_moves_norm': [],
        'num_opponent_moves': [],
        'num_opponent_moves_norm': [],
        'delta_moves': [],
        'delta_moves_norm': [],
        'num_open_fields': [],
        'num_open_fields_norm': [],
        'num_fields': [],
        'position_row': [],
        'position_row_norm': [],
        'position_col': [],
        'position_col_norm': [],
        'delta_center': [],
        'delta_center_norm': [],
        'game_won': []
    });

    game_count = 10000
    for i in range(0, game_count):
        # Create a new dataset for the game.
        num_moves_made = []
        num_available_moves = []
        num_available_moves_norm = []
        num_opponent_moves = []
        num_opponent_moves_norm = []
        delta_moves = []
        delta_moves_norm = []
        num_open_fields = []
        num_open_fields_norm = []
        num_fields = []
        position_row = []
        position_row_norm = []
        position_col = []
        position_col_norm = []
        game_won = []
        delta_center = []
        delta_center_norm = []

        # Initialize a new game.
        game = isolation.Board(player, opponent)
        counter = 0

        # Play a game with random moves and record the data.
        while True:
            legal_player_moves = game.get_legal_moves()

            # If the active player is player, then record the current game state.
            if game._active_player == player and game.get_player_location(player) is not None:
                len_opponent_moves = len(game.get_legal_moves(game._inactive_player))
                num_moves_made.append(counter)
                num_fields.append(game.height * game.width)
                open_field_count = len([1 for field in game._board_state[0:-3] if field is game.BLANK])
                num_open_fields.append(open_field_count)
                num_open_fields_norm.append(open_field_count / (game.height * game.width))
                num_available_moves.append(len(legal_player_moves))
                num_available_moves_norm.append(len(legal_player_moves) / open_field_count)
                num_opponent_moves.append(len_opponent_moves)
                num_opponent_moves_norm.append(len_opponent_moves / open_field_count)
                delta_moves.append(len(legal_player_moves) - len_opponent_moves)
                delta_moves_norm.append((len(legal_player_moves) - len_opponent_moves) / open_field_count)
                position = game.get_player_location(player)
                position_row.append(-1 if position is None else position[0])
                position_row_norm.append(-1 if position is None else position[0] / game.height)
                position_col.append(-1 if position is None else position[1])
                position_col_norm.append(-1 if position is None else position[1] / game.width)

                delta_x = position[0] - game.height / 2
                delta_y = position[1] - game.width / 2
                delta_center.append(delta_x * delta_x + delta_y * delta_y)
                delta_center_norm.append(delta_x * delta_x / (game.height * game.height) + delta_y * delta_y / (game.width * game.width))
            counter += 1

            curr_move = game._active_player.get_move(game, lambda x: 10000)

            if curr_move is None:
                curr_move = Board.NOT_MOVED

            if curr_move not in legal_player_moves:
                game_won = [1 if game._inactive_player == player else 0 for i in range(0, len(num_moves_made))]
                samples = samples.append(pd.DataFrame({
                    'num_moves_made': num_moves_made,
                    'num_available_moves': num_available_moves,
                    'num_available_moves_norm': num_available_moves_norm,
                    'num_opponent_moves': num_opponent_moves,
                    'num_opponent_moves_norm': num_opponent_moves_norm,
                    'delta_moves': delta_moves,
                    'delta_moves_norm': delta_moves_norm,
                    'num_open_fields': num_open_fields,
                    'num_open_fields_norm': num_open_fields_norm,
                    'num_fields': num_fields,
                    'position_row': position_row,
                    'position_row_norm': position_row_norm,
                    'position_col': position_col,
                    'position_col_norm': position_col_norm,
                    'delta_center': delta_center,
                    'delta_center_norm': delta_center_norm,
                    'game_won': game_won
                }));
                break;

            game.apply_move(curr_move)
        print('{}/{}'.format(i, game_count))
    # print(samples.head())

    corr = samples.drop(labels='game_won', axis=1).corrwith(samples['game_won'])
    print(corr)
    # plot_corr(samples)

if __name__ == '__main__':
    # createOpeningBook(custom_score)
    recordGameSamples()
