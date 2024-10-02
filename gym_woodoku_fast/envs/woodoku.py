import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .utils import get_three_random_blocks, BLOCK_MATRIX, VALID_MATRIX

class WoodokuFastEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(243)
        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(low=0, high=1, shape=(9, 9), dtype=np.int8),
                "block_1": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_2": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
                "block_3": spaces.Box(low=0, high=1, shape=(5, 5), dtype=np.int8),
            }
        )

    def reset(self):
        # Board & Blocks
        self.board = np.zeros((9, 9), dtype=np.int8)
        self.blocks, self.valid_block = get_three_random_blocks(BLOCK_MATRIX, VALID_MATRIX)
        self.valid_action = self.valid_block

        # Score & Meta
        self.score = 0
        self.combo = 0
        self.straight = 0
        self.n_cell = 0
        self.is_legal = 0
        self.terminated = 0

        return self._get_obs(), self._get_info()

    def _get_obs(self):
        return {
            "board": self.board,
            "block_1": self.blocks[0 * 81 + 20][:5, :5],
            "block_2": self.blocks[1 * 81 + 20][:5, :5],
            "block_3": self.blocks[2 * 81 + 20][:5, :5],
        }

    def _get_info(self):
        return {
            'action_mask': self.valid_action,
            'score': self.score,
            'straight': self.straight,
            'combo': self.combo,
            'is_legal': self.is_legal,
            'n_cell': self.n_cell
        }

    def step(self, action):
        if not self.valid_action[action]:
            self.is_legal = 0
            return self._get_obs(), None, self.terminated, False, self._get_info()

        # Make action
        self.is_legal = 1
        self.board = self.board + self.blocks[action]
        self.n_cell = self.blocks[action].sum()
        self._disable_block(action // 81)

        # Burning
        reward = self._burn()
        self.score += reward

        # Check blocks persistance
        if self.valid_block.sum() == 0:
            self.blocks, self.valid_block = get_three_random_blocks(BLOCK_MATRIX, VALID_MATRIX)

        # Compute valid actions
        board_matrix = np.tile(self.board[None, :, :], reps=(243, 1, 1))
        step_matrix = board_matrix + self.blocks  # (243, 9, 9)

        ## Validation
        self.valid_action = (
                self.valid_block &
                (step_matrix.max(axis=(-1, -2)) < 2)
        )

        # Other
        self.terminated = self.valid_action.sum() == 0

        return self._get_obs(), reward, self.terminated, False, self._get_info()

    def _disable_block(self, block_ind):
        self.blocks[block_ind * 81: (block_ind + 1) * 81] = 0
        self.valid_block[block_ind * 81: (block_ind + 1) * 81] = 0

    def _burn(self):
        # Conditions
        burn_columns = (self.board.sum(axis=0) == 9)
        burn_rows = (self.board.sum(axis=1) == 9)
        boxes = self.board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(-1, 3, 3)  # Shape: (9, 3, 3)
        burn_boxes = (boxes.sum(axis=(1, 2)) == 9)

        # Burning
        self.board[burn_rows, :] = 0
        self.board[:, burn_columns] = 0

        reshaped_board = self.board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(-1, 3, 3)
        reshaped_board[burn_boxes] = 0
        self.board = reshaped_board.reshape(3, 3, 3, 3).swapaxes(1, 2).reshape(9, 9)
        return self._get_score(burn_rows, burn_columns, burn_boxes)

    def _get_score(self, burn_rows, burn_columns, burn_boxes):
        self.combo = burn_rows.sum() + burn_columns.sum() + burn_boxes.sum()
        if self.combo > 0:
            self.straight += 1
        else:
            self.straight = 0

        if self.combo == 0:
            return self.n_cell
        else:
            return 28 * self.combo + 10 * self.straight + self.n_cell - 20