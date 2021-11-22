from collections import Counter, deque
from collections import OrderedDict
import numpy as np
import sys, pygame

AI_VS_AI = False # set to True to let the AI play against itself
START_PLAYER = 0 # 1 is AI, 0 is user
DEPTH = 2 # increase to have more precise AI, but longer computational time

NUM_COLUMNS = 7
COLUMN_HEIGHT = 6
FOUR = 4
MAX_TABLE_SIZE = 10000

def valid_moves(board):
    moves = []
    TOP = 0b1000000_1000000_1000000_1000000_1000000_1000000_1000000
    for i in range(NUM_COLUMNS):
        if TOP & (1 << board[2][i]) == 0: # shifting height by one should not end up in the top row if not full
            moves.append(i)
    return moves

def play(board, col, player):
    move = 1 << board[2][col] # left shift 1 of height[col] positions. es height[col]=2 -> 100 (the 1 corresponds to the first free element in a column)
    board[2][col] += 1
    board[player] ^= move

def four_in_a_row(bitboard):
    directions = [1, 7, 6, 8] # | - \ /
    for dir in directions:
        temp = bitboard & (bitboard >> dir)
        if temp & (temp >> 2 * dir):
            return True
    return False

def take_back(board, col, player):
    board[2][col] -= 1
    move = 1 << board[2][col]
    board[player] ^= move

def _mc(board, player):
    p = int(not player)
    moves = deque()
    found = False
    while valid_moves(board):
        p = int(not p)
        c = np.random.choice(valid_moves(board))
        play(board, c, p)
        moves.append((c, p))
        if four_in_a_row(board[p]):
            found = True
            break
    while moves:
        col, player = moves.pop()
        take_back(board, col, player)
    if found:
        return p
    else:
        return -1 # draw

def montecarlo(board, player):
    montecarlo_samples = 100
    cnt = Counter(_mc((board), player) for _ in range(montecarlo_samples))
    return (cnt[player] - cnt[int(not player)]) / montecarlo_samples 

def eval_board(board, player):
    if four_in_a_row(board[player]):
        return 1
    elif four_in_a_row(board[int(not player)]):
        return -1
    else:
        # Not terminal, let's simulate...
        return montecarlo(board, player)

# To increase pruning chances I explore first the center columns, which are better than edge columns on average
def reorder(moves):
    ideal = np.array([3,4,2,1,5,6,0])
    return ideal[np.in1d(ideal, moves, assume_unique=True)]

def mandatory_move(moves, board, player): # immediate loss or win
    for ply in moves:
        play(board, ply, int(not player))
        if four_in_a_row(board[int(not player)]):
            take_back(board, ply, int(not player))
            return ply
        take_back(board, ply, int(not player))
        play(board, ply, player)
        if four_in_a_row(board[player]): # win move
            take_back(board, ply, player)
            return ply
        take_back(board, ply, player)
    return -1


def alpha_beta(board, player, depth, alpha, beta, table):
    possible = reorder(valid_moves(board))
    key = str(board[player]) + str(board[int(not player)])
    if key in table:
        return table[key]    
    if depth == 0 or not any(possible):
        return None, eval_board(board, player)
    imp_move = mandatory_move(possible, board, player)
    if imp_move != -1:
        return (imp_move, 1)
    max_eval = (None, -1)
    for ply in possible:
        play(board, ply, player)
        _, val = alpha_beta(board, int(not player), depth - 1, -beta, -alpha, table)
        val = -val
        take_back(board, ply, player)
        if max_eval[0] is None or val > max_eval[1]:
            max_eval = (ply, val)
        alpha = max(alpha, val)
        if beta <= alpha:
            break
    table[key] = max_eval
    if len(table) > MAX_TABLE_SIZE:
        [table.popitem(last=False) for _ in range(100)] # remove oldest 100 items      
    return max_eval

pygame.init()
pygame.display.set_caption('Connect 4')
size = width, height = (800, 600)
board_size = (637,550)
screen = pygame.display.set_mode(size)
board_img = pygame.image.load('board.png')
board_img = pygame.transform.smoothscale(board_img, board_size)
boardrect = board_img.get_rect()
white = (255, 255, 255)
red = (230, 30, 30)
light_red = (255, 153, 153)
yellow = (234, 204, 73)

step = 86
radius = 40
start_pos = (144, 86)

def draw(board, click=False):
    # convert the bitboard to 2d array
    top = {6,13,20,27,34,41}
    bits0 = [(board[0] >> i) & 1 for i in range(0,48) if i not in top]
    bits1 = [-((board[1] >> i) & 1) for i in range(0,48) if i not in top]
    board = np.add(np.reshape(bits0, (7,6)), np.reshape(bits1, (7,6)))

    x_mouse = (pygame.mouse.get_pos()[0] - start_pos[0] + step/2) // step
    screen.fill(white)
    first = True
    for j in range(board.shape[1]):
        for i in range(board.shape[0]):
            if board[i][j] == 1:
                pygame.draw.circle(screen, red, (start_pos[0] + i*step, start_pos[1] + (5-j)*step), radius)
            elif board[i][j] == -1:
                pygame.draw.circle(screen, yellow, (start_pos[0] + i*step, start_pos[1] + (5-j)*step), radius)

            if not AI_VS_AI and not click and x_mouse == i and board[i][j] == 0 and first:
                pygame.draw.circle(screen, light_red, (start_pos[0] + i*step, start_pos[1] + (5-j)*step), radius)
                first = False
    screen.blit(board_img, (81,25))
    pygame.display.flip()

board = [0,0, [0,7,14,21,28,35,42]] # [p1, p2, height]
table = OrderedDict() # ordered dict so I can remove the LRU configuration
player = START_PLAYER

finish = False
draw(board)

first_move = True
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.MOUSEBUTTONDOWN and AI_VS_AI == False and not finish:
            left = pygame.mouse.get_pressed()[0]
            if left and player == 0:
                col = (pygame.mouse.get_pos()[0] - start_pos[0] + step/2) // step
                if col in valid_moves(board):
                    play(board, int(col), player)
                    player = int(not player)
                    draw(board, click=True)

    if (AI_VS_AI or player == 1) and not finish:
        best_move = None
        if first_move:
            best_move = 3 # middle column
            first_move = False
        else:
            best_move, _ = alpha_beta(board, player, DEPTH, -np.inf, np.inf, table)
        if best_move is None:
            print(f"Draw")
            finish = True
        else:
            play(board, best_move, player)
            if four_in_a_row(board[0]):
                print(f"Red won")
                finish = True
            if four_in_a_row(board[1]):
                print(f"Yellow won")
                finish = True
            player = int(not player)
    draw(board)


# Montecarlo Tree Search  - implemented but not used, works on professor's data structure (2d array for the board)
'''
def UCB(val, n, N):
    return val + 2*np.sqrt(np.log(N)/n)

class Node:
    def __init__(self, t, n, parent, move):
        self.t = t
        self.n = n
        self.parent = parent
        self.move = move

# works with old data structure where board is a 2d array
def mcts(visited, board, player, current_player):
    possibilities = valid_moves(board)
    if not possibilities:
        return
    next = None
    nodep = visited[board.tobytes()]
    for ply in possibilities:
        play(board, ply, current_player)
        if board.tobytes() not in visited:
            next = np.copy(board)
            take_back(board, ply)
            break
        elif next is None or UCB(visited[board.tobytes()].t, visited[board.tobytes()].n, nodep.n) > UCB(visited[next.tobytes()].t, visited[next.tobytes()].n, nodep.n):
            next = np.copy(board)
        take_back(board, ply)
    if next.tobytes() not in visited: # rollout
        val = _mc(np.copy(next), current_player)
        if player == -1:
            val = -val
        visited[next.tobytes()] = Node(val, 1, board.tobytes(), ply)
        # backpropagation
        next = next.tobytes()
        while visited[next].parent is not None:
            parent = visited[next].parent
            visited[parent].t += val
            visited[parent].n += 1
            next = parent
    else: # expansion
        mcts(visited, next, player, -player)

def MCTS(board, player, iterations):
    visited = dict()
    visited[board.tobytes()] = Node(0, 0, None, 0)
    nodep = visited[board.tobytes()]
    for _ in range(iterations):
        mcts(visited, np.copy(board), player, player)
    best = None
    for ply in valid_moves(board):
        play(board, ply, player)
        next = visited[board.tobytes()]
        #print(f"col: {next.move}, val: {next.t}, n: {next.n}")
        if best is None or UCB(next.t, next.n, nodep.n) > UCB(best.t, best.n, nodep.n):
            best = next
        take_back(board, ply)
    return best.move
'''