from ..settings import *

def piece_to_one_hot(piece_type):
    # Converts a piece type string to a one-hot vector
    vector = [0] * len(POSSIBLE_KEYS)
    if piece_type in PIECE_TYPE2NUM:
        vector[PIECE_TYPE2NUM[piece_type] - 1] = 1
    return vector

def get_features(grid):

    heights = get_column_heights(grid)
    diff = get_column_height_diff(heights)
    holes = get_holes(grid)

    return (heights, diff, holes)

def get_column_heights(grid):
    heights = [0] * GRID_WIDTH

    if not grid:
        return heights

    for j in range(GRID_WIDTH):
        for i in range(GRID_DEPTH):
            if int(grid[i][j]) > 0:
                heights[j] = GRID_DEPTH - i 
                break 

    return heights

def get_column_height_diff(heights):
    diff = [0] * (len(heights) - 1)
    
    for i in range(len(heights)):
        if i == 0:
            diff[i] = abs(heights[i+1] - heights[i])
        elif i == len(heights) - 1:
            diff[i] = abs(heights[i] - heights[i-1])
        else:
            diff[i] = max(abs(heights[i+1] - heights[i]), abs(heights[i] - heights[i-1]))

    return diff

def get_holes(grid):
    holes = [0] * GRID_WIDTH

    if not grid:
        return holes

    for j in range(GRID_WIDTH):
        occupied = False
        for i in range(GRID_DEPTH):
            holes_count = 0
            if int(grid[i][j]) > 0:
                occupied = True
            if int(grid[i][j]) == 0 and occupied:
                holes_count += 1
            
            holes[j] = holes_count

    return holes   