import torch.nn as nn


from chess import Board, pgn
from auxillary_func import board_to_matrix
import torch
from model import ChessModel
import pickle
import numpy as np
from  bridge import GameStateConverter
gameStateConverter = GameStateConverter()
    

def load_model_and_predict_move(board, game_state):

    with open("models/heavy_move_to_int", "rb") as file:
        move_to_int = pickle.load(file)

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Load the model
    model = ChessModel(num_classes=len(move_to_int))
    model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth"))
    model.to(device)




    int_to_move = {v: k for k, v in move_to_int.items()}
    # Function to make predictions
    def predict_move(board):
        X_tensor = prepare_input(board).to(device)
        
        with torch.no_grad():
            logits = model(X_tensor)
        
        logits = logits.squeeze(0)  # Remove batch dimension
        
        probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
        legal_moves = list(gameStateConverter.get_legal_moves_uci(game_state))
        print("All the legal moves",legal_moves)
        legal_moves_uci = [move.uci() for move in legal_moves]
        sorted_indices = np.argsort(probabilities)[::-1]
        for move_index in sorted_indices:
            move = int_to_move[move_index]
            if move in legal_moves_uci:
                return move
        
        return None

    # Predict and make a move
    best_move = predict_move(board)
    return best_move



def prepare_input(board:Board):
    matrix = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    X_tensor = X_tensor.reshape((1,13,8,8))
    print(X_tensor.shape)
    return X_tensor 

