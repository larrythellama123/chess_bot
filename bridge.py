import chess
import numpy as np
import torch
from piece import pieces  # Your existing piece module

Piece = pieces()

class GameStateConverter:
    """Converts between your custom gamestate and standard chess representations"""
    
    def __init__(self):
        # Mapping from your piece system to standard chess
        self.piece_type_map = {
            Piece.pawn: chess.PAWN,
            Piece.knight: chess.KNIGHT,
            Piece.bishop: chess.BISHOP,
            Piece.rook: chess.ROOK,
            Piece.queen: chess.QUEEN,
            Piece.king: chess.KING
        }
        
        # Reverse mapping
        self.reverse_piece_map = {v: k for k, v in self.piece_type_map.items()}


    def gamestate_to_chess_board(self, game_state):
        board = chess.Board()  # Empty board
        board.clear()
        
        # Convert your board to standard format
        for row in range(8):
            for col in range(8):
                piece_value = game_state.board[row][col]
                
                if piece_value != 0:  # If there's a piece
                    # Extract piece type and color from your encoding
                    piece_type = self._extract_piece_type(piece_value)
                    color = self._extract_color(piece_value)
                    
                    if piece_type in self.piece_type_map:
                        chess_piece_type = self.piece_type_map[piece_type]
                        chess_color = chess.WHITE if color == Piece.white else chess.BLACK
                        
                        # Convert row,col to chess square (a1=0, h8=63)
                        square = row * 8 + col
                        
                        # Create and place piece
                        chess_piece = chess.Piece(chess_piece_type, chess_color)
                        board.set_piece_at(square, chess_piece)
        
        # Set game state properties
        board.turn = chess.WHITE if game_state.current_color == Piece.white else chess.BLACK
        
        # Set castling rights (you'll need to track these in your gamestate)
        self._set_castling_rights(board, game_state)
        
        return board
    
    def chess_board_to_gamestate_tensor(self, chess_board):
        """Convert chess.Board directly to neural network tensor"""
        features = []
        
        # 1. Piece positions (8x8x12 = 768 features)
        piece_features = np.zeros((8, 8, 12))
        
        piece_map = {
            (chess.PAWN, chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK, chess.WHITE): 3,
            (chess.QUEEN, chess.WHITE): 4,
            (chess.KING, chess.WHITE): 5,
            (chess.PAWN, chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK, chess.BLACK): 9,
            (chess.QUEEN, chess.BLACK): 10,
            (chess.KING, chess.BLACK): 11,
        }
        
        for square in chess.SQUARES:
            piece = chess_board.piece_at(square)
            if piece:
                row = square // 8
                col = square % 8
                piece_idx = piece_map[(piece.piece_type, piece.color)]
                piece_features[row, col, piece_idx] = 1
        
        features.extend(piece_features.flatten())
        
        # 2. Additional features (5 features)
        features.append(1 if chess_board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if chess_board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if chess_board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if chess_board.has_queenside_castling_rights(chess.BLACK) else 0)
        features.append(1 if chess_board.turn == chess.WHITE else 0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def gamestate_to_tensor(self, game_state):
        """Convert your gamestate directly to neural network input"""
        # Method 1: Convert via chess.Board (more reliable)
        chess_board = self.gamestate_to_chess_board(game_state)
        return self.chess_board_to_gamestate_tensor(chess_board)
        
        # Method 2: Direct conversion (faster but needs careful implementation)
        # return self._direct_gamestate_to_tensor(game_state)
    
    def _direct_gamestate_to_tensor(self, game_state):
        """Direct conversion without intermediate chess.Board"""
        features = []
        
        # 1. Piece positions (8x8x12 = 768 features)
        piece_features = np.zeros((8, 8, 12))
        
        # Map your pieces to neural network indices
        your_piece_map = {
            (Piece.pawn, Piece.white): 0,
            (Piece.knight, Piece.white): 1,
            (Piece.bishop, Piece.white): 2,
            (Piece.rook, Piece.white): 3,
            (Piece.queen, Piece.white): 4,
            (Piece.king, Piece.white): 5,
            (Piece.pawn, Piece.black): 6,
            (Piece.knight, Piece.black): 7,
            (Piece.bishop, Piece.black): 8,
            (Piece.rook, Piece.black): 9,
            (Piece.queen, Piece.black): 10,
            (Piece.king, Piece.black): 11,
        }
        
        for row in range(8):
            for col in range(8):
                piece_value = game_state.board[row][col]
                if piece_value != 0:
                    piece_type = self._extract_piece_type(piece_value)
                    color = self._extract_color(piece_value)
                    
                    if (piece_type, color) in your_piece_map:
                        piece_idx = your_piece_map[(piece_type, color)]
                        piece_features[row, col, piece_idx] = 1
        
        features.extend(piece_features.flatten())
        
        # 2. Additional features - you'll need to add these to your gamestate
        features.append(1 if game_state.white_can_castle_kingside else 0)
        features.append(1 if game_state.white_can_castle_queenside else 0)
        features.append(1 if game_state.black_can_castle_kingside else 0)
        features.append(1 if game_state.black_can_castle_queenside else 0)
        features.append(1 if game_state.current_color == Piece.white else 0)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_piece_type(self, piece_value):
        """Extract piece type from your encoding"""
        # You'll need to implement this based on your piece encoding
        # This assumes you use bitwise operations like: piece_value & Piece.type_mask
        
        if Piece.is_type(piece_value, Piece.pawn):
            return Piece.pawn
        elif Piece.is_type(piece_value, Piece.knight):
            return Piece.knight
        elif Piece.is_type(piece_value, Piece.bishop):
            return Piece.bishop
        elif Piece.is_type(piece_value, Piece.rook):
            return Piece.rook
        elif Piece.is_type(piece_value, Piece.queen):
            return Piece.queen
        elif Piece.is_type(piece_value, Piece.king):
            return Piece.king
        
        return None
    
    def _extract_color(self, piece_value):
        """Extract color from your encoding"""
        # You'll need to implement this based on your piece encoding
        if Piece.is_color(piece_value, Piece.white, True):
            return Piece.white
        else:
            return Piece.black
    
    def _set_castling_rights(self, chess_board, game_state):
        """Set castling rights on chess board"""
        # You'll need to track castling rights in your gamestate
        # For now, make educated guesses based on king/rook positions
        
        # Check if kings and rooks are in starting positions
        white_king_pos = None
        black_king_pos = None
        
        for row, col in game_state.white_positions:
            if Piece.is_type(game_state.board[row][col], Piece.king):
                white_king_pos = (row, col)
                break
        
        for row, col in game_state.black_positions:
            if Piece.is_type(game_state.board[row][col], Piece.king):
                black_king_pos = (row, col)
                break
        
        # Simple heuristic: allow castling if king is on starting square
        if white_king_pos == (7, 4):  # White king on e1
            chess_board.set_castling_fen("KQkq")  # Allow all castling
        else:
            chess_board.set_castling_fen("-")  # No castling

# Enhanced Neural Network that works with your gamestate
class GameStateChessNet(torch.nn.Module):
    """Neural network adapted for your gamestate format"""
    
    def __init__(self, input_size=773):
        super(GameStateChessNet, self).__init__()
        
        self.converter = GameStateConverter()
        
        # Same architecture as before
        self.fc1 = torch.nn.Linear(input_size, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc_eval = torch.nn.Linear(64, 1)
        self.dropout = torch.nn.Dropout(0.3)
        
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc3(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc4(x))
        
        eval_output = torch.tanh(self.fc_eval(x))
        return eval_output
    
    def evaluate_gamestate(self, game_state):
        """Evaluate your gamestate directly"""
        with torch.no_grad():
            position_tensor = self.converter.gamestate_to_tensor(game_state)
            evaluation = self.forward(position_tensor.unsqueeze(0))
            return evaluation.item()

# Usage example
def integrate_with_your_system():
    """Example of how to integrate with your existing system"""
    
    # 1. Train the network using PGN data (standard format)
    model = GameStateChessNet()
    
    # 2. Train using the original chess training code
    # (This part uses standard chess.Board format)
    
    # 3. Use the trained model with your gamestate
    def evaluate_your_position(game_state):
        """Evaluate your custom gamestate"""
        return model.evaluate_gamestate(game_state)
    
    # 4. Integrate with your existing evaluation
    def enhanced_evaluate(game_state):
        """Combine neural network with your hand-crafted evaluation"""
        
        # Your existing evaluation
        traditional_eval = your_enhanced_evaluation.evaluate(game_state)
        
        # Neural network evaluation
        nn_eval = model.evaluate_gamestate(game_state) * 1000  # Scale to centipawns
        
        # Combine them (you can adjust weights)
        combined_eval = 0.7 * traditional_eval + 0.3 * nn_eval
        
        return combined_eval
    
    return evaluate_your_position, enhanced_evaluate

# What you need to add to your gamestate class
class EnhancedGameState:
    """Your gamestate with additional neural network support"""
    
    def __init__(self):
        # Your existing gamestate properties
        self.board = None
        self.current_color = None
        self.white_positions = []
        self.black_positions = []
        
        # Add these for neural network compatibility
        self.white_can_castle_kingside = True
        self.white_can_castle_queenside = True
        self.black_can_castle_kingside = True
        self.black_can_castle_queenside = True
        self.en_passant_square = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        
        # Neural network converter
        self.converter = GameStateConverter()
    
    def to_neural_input(self):
        """Convert this gamestate to neural network input"""
        return self.converter.gamestate_to_tensor(self)
    
    def evaluate_with_nn(self, neural_model):
        """Evaluate this position using neural network"""
        return neural_model.evaluate_gamestate(self)