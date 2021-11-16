from collections import Counter
import numpy as np
import sys, pygame

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4

# Board can be initiatilized with `board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)`
# Notez Bien: Connect 4 "columns" are actually NumPy "rows"

def valid_moves(board):
    """Returns columns where a disc may be played"""
    return [n for n in range(NUM_COLUMNS) if board[n, COLUMN_HEIGHT - 1] == 0]


def play(board, column, player):
    """Updates `board` as `player` drops a disc in `column`"""
    (index,) = next((i for i, v in np.ndenumerate(board[column]) if v == 0)) # index of first element with value 0 in the column
    board[column, index] = player


def take_back(board, column):
    """Updates `board` removing top disc from `column`"""
    (index,) = [i for i, v in np.ndenumerate(board[column]) if v != 0][-1] # index of last non-zero element in the column
    board[column, index] = 0


def four_in_a_row(board, player):
    """Checks if `player` has a 4-piece line"""
    return (
        any(
            all(board[c, r] == player)
            for c in range(NUM_COLUMNS)
            for r in (list(range(n, n + FOUR)) for n in range(COLUMN_HEIGHT - FOUR + 1))
        )
        or any(
            all(board[c, r] == player)
            for r in range(COLUMN_HEIGHT)
            for c in (list(range(n, n + FOUR)) for n in range(NUM_COLUMNS - FOUR + 1))
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co, co + FOUR))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
        or any(
            np.all(board[diag] == player)
            for diag in (
                (range(ro, ro + FOUR), range(co + FOUR - 1, co - 1, -1))
                for ro in range(0, NUM_COLUMNS - FOUR + 1)
                for co in range(0, COLUMN_HEIGHT - FOUR + 1)
            )
        )
    )

## Montecarlo evalutaion
def _mc(board, player):
    p = -player
    while valid_moves(board):
        p = -p
        c = np.random.choice(valid_moves(board))
        play(board, c, p)
        if four_in_a_row(board, p):
            return p
    return 0


def montecarlo(board, player):
    montecarlo_samples = 100
    cnt = Counter(_mc(np.copy(board), player) for _ in range(montecarlo_samples))
    return (cnt[1] - cnt[-1]) / montecarlo_samples # score representing who is probably going to win (-1 is a certain win for player -1, 0 is 50%, 1 is 100% win for player 1)

def my_montecarlo(board, player):
    montecarlo_samples = 100
    cnt = Counter(_mc(np.copy(board), player) for _ in range(montecarlo_samples))
    return (cnt[player] - cnt[-player]) / montecarlo_samples # score representing if player is probably going to win (-1 is a certain win for opponent, 0 is 50%, 1 is 100% win for player)


def eval_board(board, player):
    if four_in_a_row(board, 1):
        # Alice won
        return 1
    elif four_in_a_row(board, -1):
        # Bob won
        return -1
    else:
        # Not terminal, let's simulate...
        return montecarlo(board, player)

def my_eval_board(board, player):
    if four_in_a_row(board, player):
        # Alice won
        return 1
    elif four_in_a_row(board, -player):
        # Bob won
        return -1
    else:
        # Not terminal, let's simulate...
        return my_montecarlo(board, player)

def alpha_beta(board, player, depth, alpha, beta):
    possible = valid_moves(board)
    if depth == 0 or not possible:
        return None, eval_board(board, player)
    
    if player == 1:
        max_eval = (None, -1)
        for ply in possible:
            play(board, ply, player)
            _, val = alpha_beta(board, -player, depth - 1, alpha, beta) #  | is the union set operator
            val = -val
            take_back(board, ply)
            if val > max_eval[1]:
                max_eval = (ply, val)
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = (None, 1)
        for ply in possible:
            play(board, ply, player)
            _, val = alpha_beta(board, -player, depth - 1, alpha, beta) #  | is the union set operator
            val = -val
            take_back(board, ply)
            if val < min_eval[1]:
                min_eval = (ply, val)
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval

def minmax(board, player, depth):
    #val = 1 if four_in_a_row(board, player) else -1 if four_in_a_row(board, -player) else 0
    possible = valid_moves(board)
    if depth == 0 or not possible:
        return None, my_eval_board(board, player)
    evaluations = list()
    for ply in possible:
        play(board, ply, player)
        _, val = minmax(board, -player, depth - 1) #  | is the union set operator
        take_back(board, ply)
        evaluations.append((ply, -val))
    return max(evaluations, key=lambda k: k[1])

board = np.zeros((NUM_COLUMNS, COLUMN_HEIGHT), dtype=np.byte)
'''
player = 1
won = False
while not won:
    col = np.random.choice(valid_moves(board))
    play(board,col,player)
    won = four_in_a_row(board, player)
    player = -player
take_back(board, col)
#print(col, -player)
player = -player
'''

pygame.init()
pygame.display.set_caption('Connect 4')
size = width, height = (800, 600)
board_size = (637,550)
screen = pygame.display.set_mode(size)
board_img = pygame.image.load('connect4assets/board.png')
board_img = pygame.transform.scale(board_img, board_size)
boardrect = board_img.get_rect()
white = (255, 255, 255)
red = (230, 30, 30)
yellow = (234, 204, 73)

step = 86
radius = 40
start_pos = (144, 86)

def draw(board):
    for j in range(board.shape[1]):
        for i in range(board.shape[0]):
            if board[i][j] == 1:
                pygame.draw.circle(screen, red, (start_pos[0] + i*step, start_pos[1] + (5-j)*step), radius)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, yellow, (start_pos[0] + i*step, start_pos[1] + (5-j)*step), radius)


def run():
    pass

play(board, 4, 1)
play(board, 0, -1)
play(board, 5, 1)
play(board, 0, -1)
play(board, 6, 1)
print(board)
player = -1

won = False
screen.fill(white)
draw(board)
screen.blit(board_img, (81,25))
pygame.display.flip()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    while not won:        
        #best_move, _ = minmax(board, player, 2)
        best_move, _ = alpha_beta(board, player, 2, -2, 2)
        play(board, best_move, player)
        won = four_in_a_row(board, player) or four_in_a_row(board, -player)
        player = -player
        if won:
            print(board)

        screen.fill(white)
        draw(board)
        screen.blit(board_img, (81,25))
        pygame.display.flip()
        

# neat con board come input, 7 neuroni output (indicano la colonna in cui piazzare). Fitness formula tiene conto della vittoria più rapida, più una penalità per ogni sconfitta credo