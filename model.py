import torch.nn as nn


from chess import Board, pgn
from auxillary_func import board_to_matrix
import torch
import pickle
import numpy as np



class ChessModel(nn.Module):
    def __init__(self, num_classes):
        super(ChessModel, self).__init__()
        # conv1 -> relu -> conv2 -> relu -> flatten -> fc1 -> relu -> fc2
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Initialize weights
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv2.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x
    


    # def predict_move(self):

    #     with open("models/heavy_move_to_int", "rb") as file:
    #         move_to_int = pickle.load(file)

    #     # Check for GPU
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f'Using device: {device}')

    #     # Load the model
    #     model = ChessModel(num_classes=len(move_to_int))
    #     model.load_state_dict(torch.load("models/TORCH_100EPOCHS.pth"))
    #     model.to(device)
    #     # model.eval()  # Set the model to evaluation mode (it may be reductant)



    #     int_to_move = {v: k for k, v in move_to_int.items()}
    #     # Function to make predictions
    #     def predict_move(board: Board):
    #         X_tensor = self.prepare_input(board).to(device)
            
    #         with torch.no_grad():
    #             logits = model(X_tensor)
            
    #         logits = logits.squeeze(0)  # Remove batch dimension
            
    #         probabilities = torch.softmax(logits, dim=0).cpu().numpy()  # Convert to probabilities
    #         legal_moves = list(GameStateConverter.get_legal_moves_uci())
    #         legal_moves_uci = [move.uci() for move in legal_moves]
    #         sorted_indices = np.argsort(probabilities)[::-1]
    #         for move_index in sorted_indices:
    #             move = int_to_move[move_index]
    #             if move in legal_moves_uci:
    #                 return move
            
    #         return None




    #     # Initialize a chess board
    #     board = Board()



    #     # Predict and make a move
    #     best_move = predict_move(board)
    #     board.push_uci(best_move)
    #     board


    #     print(str(pgn.Game.from_board(board)))


    # def prepare_input(self,board:Board):
    #     matrix = board_to_matrix(board)
    #     X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0)
    #     X_tensor = X_tensor.reshape((1,13,8,8))
    #     print(X_tensor.shape)
    #     return X_tensor


        