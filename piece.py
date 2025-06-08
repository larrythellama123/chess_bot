"""specifies the type of piece this is
    the movement range of this piece
    generates the attack map and stores it
"""

class pieces:
    def __init__(self):
        #3 bits on the right tell what type of piece it is
        self.pawn = 1
        self.rook = 2
        self.knight = 3
        self.bishop = 4
        self.queen = 5
        self.king  = 6
        #2 bits on the left decide the colour of the piece
        self.white = 8
        self.black = 16

    def is_type(self, piece,target_piece):
        piece_type = piece & 7
        if piece_type == target_piece:
            return True
        else:
            return False
        
    def is_color(self, piece,initial_piece,same_color=True):
        if piece == 0 or initial_piece == 0:
            return False
        piece_color = piece & 24
        initial_piece_color  = initial_piece & 24
        if same_color:
            if piece_color == initial_piece_color:
                return True
            else:
                return False
        else:
            if piece_color == initial_piece_color:
                return False
            else:
                return True
         
    def get_piece_color(self, piece):
        piece_color = piece & 24
        if piece_color == self.white:
            return self.white
        else:
            return self.black


        
        