import numpy as np
from chess import Board, SQUARES

piece_map = {
        'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5,
        'P': 6, 'N': 7, 'B': 8, 'R': 9, 'Q': 10, 'K': 11,
    }

def board_to_matrix(board:Board):
    #he adds 13the board for legal moves?
    matrix = np.zeros((8, 8, 13), dtype=np.int8)
    for square in SQUARES:
        piece = board.piece_at(square)
        if piece:
            row, col = divmod(square, 8)
            matrix[7 - row, col, piece_map[piece.symbol()]] = 1
    return matrix

def create_input_for_nn(games):
    X = []
    y = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            X.append(board_to_matrix(board))
            y.append(move.uci())
            board.push(move)
    return np.array(X, dtype=np.float32), np.array(y)


def encode_moves(moves):
    move_to_int = {move: idx for idx, move in enumerate(set(moves))}
    return np.array([move_to_int[move] for move in moves], dtype=np.float32), move_to_int

