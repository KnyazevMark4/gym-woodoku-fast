import numpy as np
from .blocks import blocks


def get_possible_locations(block: np.array) -> np.array:
    """
    Parameters
    ----------
    block : np.array
        Array with shape (5, 5) representing particular woodoku block.

    Returns
    -------
    possible_locations : np.array
        An array with shape (81, 9, 9), where 81 is the number of possible locations and (9, 9) is the size of the board.
    valid_mask: np.array
        An array with shape (243) which includes indictor whether the action is valid.

    Usage:
        possible_locations, valid_mask = get_possible_locations(block)
    """
    possible_locations = []
    valid_mask = []
    for row_i in range(9):
        for columns_j in range(9):
            empty_board = np.zeros((13, 13))
            empty_board[row_i:row_i + 5, columns_j:columns_j + 5] = block
            possible_locations.append(empty_board[2:-2, 2:-2])
            if empty_board.sum() == empty_board[2:-2, 2:-2].sum():
                is_valid = 1
            else:
                is_valid = 0
            valid_mask.append(is_valid)
    return np.array(possible_locations), np.array(valid_mask)


def get_extended_blocks(blocks):
    """
    Function to create array with possible locations of each block.

    Parameters
    ----------
    blocks : np.array
        Array with shape (n_blocks, 5, 5) representing different woodoku blocks.

    Returns
    -------
    extended_matrix : np.array
        An array with shape (n_blocks, 81, 9, 9), where 81 is the number of possible locations and (9, 9) size of the board.
    valid_mask: ...
        ...
    """
    matrices = []
    valid_mask = []
    for block_i in blocks:
        possible_locations_i, valid_mask_i = get_possible_locations(block_i)
        matrices.append(possible_locations_i)
        valid_mask.append(valid_mask_i)
    return np.array(matrices), np.array(valid_mask)


def get_three_random_blocks(BLOCK_MATRIX, VALID_MATRIX):
    """
    Usage:
        blocks = get_three_random_blocks(BLOCK_MATRIX)
    """
    block_ids = np.random.choice(range(BLOCK_MATRIX.shape[0]), 3, replace=False)
    random_blocks = BLOCK_MATRIX[block_ids]
    valid_mask = VALID_MATRIX[block_ids]
    return random_blocks.reshape(-1, 9, 9), valid_mask.reshape(-1)


BLOCK_MATRIX, VALID_MATRIX = get_extended_blocks(blocks['woodoku'])
BLOCK_MATRIX = BLOCK_MATRIX.astype(np.int8)
VALID_MATRIX = VALID_MATRIX.astype(np.int8)
