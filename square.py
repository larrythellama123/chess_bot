"""checks if the square is in range of the piece movement
    check if there is an enemy or a team member on the same square
"""
class squares:
    def __init__(self):
        self.moved  = False
        self.white_pawn_original_location = [(6,0),(6,1),(6,2),(6,3),(6,4),(6,5),(6,6),(6,7)]
        self.black_pawn_original_location = [(1,0),(1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7)]
        