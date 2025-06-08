from piece import pieces
from chess_engine import GameState

class Eval:
    Pawn = 1
    Knight = 3
    Bishop = 3
    Rook = 5
    Queen = 9
    Piece = pieces()
    gameState = GameState()
    points = 0

    def check_pawn_struct(board):
        check_indiv_pawns(board)
        check_group_pawns(board)



    def check_indiv_pawns(attack_squares,board, row,col):
        if (row,col) in attack_squares:
            points -= 4
        if (row,col) not in defended_squares:
            """
        Identifies pawn groups (adjacent pawns of the same color) on a chess board.
        """
        pawn_groups = {}
        next_group_id = 1

        # 1. Iterate through all squares on the board
        for row in range(8):
            for row in range(8):
                Piece = board[row][col]
                # 2. Check if it's a pawn and of a specific color (e.g., White)
                if Piece and Piece.piece_type == Piece.pawn and Piece.color == Piece.white:
                    # 3. Check if the pawn is already part of a group
                    is_group_member = False
                    for group_id, squares in pawn_groups.items():
                        if square in squares:
                            is_group_member = True
                            break

                    if not is_group_member:
                        # 4. If not already in a group, start a new group
                        pawn_groups[next_group_id] = {square}
                        next_group_id += 1

                        # 5. Check adjacent squares for other pawns
                        for adj_square in chess.get_adjacent_squares(square):
                            adj_piece = board.piece_on(adj_square)

                            # Check if it's a pawn of the same color and not already in a group
                            if adj_piece and adj_piece.piece_type == chess.PAWN and \
                            adj_piece.color == chess.WHITE and not is_group_member:
                                # 6. Add the adjacent pawn to the current group
                                if next_group_id - 1 in pawn_groups:
                                    pawn_groups[next_group_id - 1].add(adj_square)

        return pawn_groups
            



    def check_group_pawns():




    








    #eval based on mobility 
    #center control
    #trapped pieces
    #king safety based on pawns
    #material   
    #pawn structs

    def evaluate(black_positions, white_positions,board,checked_path,current_color):
        #check the total value of pieces
        for white in range(white_positions):
            row,col = white
            if Piece.is_type(board[row][col], Piece.pawn):
                points += Pawn
            if Piece.is_type(board[row][col], Piece.knight):
                points += Knight
            if Piece.is_type(board[row][col], Piece.bishop):
                points += Bishop

            if Piece.is_type(board[row][col], Piece.queen):
                points += Queen

            if Piece.is_type(board[row][col], Piece.rook):
                points += Rook

        for black in range(black_positions):
            row,col = black
            if Piece.is_type(board[row][col], Piece.pawn):
                points -= Pawn
            if Piece.is_type(board[row][col], Piece.knight):
                points -= Knight
            if Piece.is_type(board[row][col], Piece.bishop):
                points -= Bishop

            if Piece.is_type(board[row][col], Piece.queen):
                points -= Queen

            if Piece.is_type(board[row][col], Piece.rook):
                points -= Rook
        




        


        
    def minmax(depth,alpha, beta, is_maximising):
        self.gameState.start_new_round()
        if depth == 0 or self.game_is_finished(board) or self.no_move(board) :
            return evaluate(board)
        if is_maximising:
            best_score = float("inf")
            for move in self.final_allowed_moves:
                Piece  = self.board[start_row][start_col]
                target_row, target_col = move.target_square
                start_row, start_col = move.start_square
                self.board[start_row][start_col] = 0
                self.board[target_row][target_col] = Piece
                score = minmax(depth-1,alpha,beta,False) 
        else:
            best_score = float("inf")
            for move in self.final_allowed_moves:
                Piece  = self.board[start_row][start_col]
                target_row, target_col = move.target_square
                start_row, start_col = move.start_square
                self.board[start_row][start_col] = 0
                self.board[target_row][target_col] = Piece
                score = minmax(depth-1,alpha,beta,False) 