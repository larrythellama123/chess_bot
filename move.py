

#move is a struct:
    #start piece
    #target piece

class Move:
    def __init__(self):
        self.start_square = (0,0)
        self.target_square = (0,0)

    def print_move(self):
        print(f"{self.start_square}")
        print(f"{self.target_square}")

