import pygame
import chess
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os

# Constants
BOARD_WIDTH = 512
BOARD_HEIGHT = 512
SQUARE_SIZE = BOARD_WIDTH // 8
FPS = 30

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT = (100, 255, 100)
LAST_MOVE = (255, 255, 0)

# Initialize pygame
pygame.init()
pygame.display.set_caption("Chess AI with Reinforcement Learning")

# Loading chess piece images
def load_pieces():
    pieces = {}
    # Black pieces keep original symbols
    black_pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    # White pieces use lowercase with '1' suffix
    white_pieces = ['p1', 'r1', 'n1', 'b1', 'q1', 'k1']
    
    # Load black pieces
    for piece in black_pieces:
        pieces[piece] = pygame.transform.scale(
            pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )
    
    # Load white pieces with modified names
    # But map them to standard chess notation for internal representation
    standard_white = ['P', 'R', 'N', 'B', 'Q', 'K']
    for i, piece in enumerate(white_pieces):
        pieces[standard_white[i]] = pygame.transform.scale(
            pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )
    
    return pieces

# Helper function to convert standard notation to our display notation
def piece_to_display_symbol(piece):
    if piece is None:
        return None
    
    symbol = piece.symbol()
    # If it's a white piece (uppercase), convert to lowercase + '1'
    if symbol.isupper():
        return symbol.lower() + '1'
    # Black pieces stay the same
    return symbol

# Neural Network for Deep Q-Learning
class ChessNN(nn.Module):
    def __init__(self):
        super(ChessNN, self).__init__()
        # Input: 8x8 board with 12 channels (6 piece types x 2 colors)
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)  # Value head - evaluates position
        self.fc3 = nn.Linear(512, 4096)  # Policy head - 64*64 possible moves (from, to)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 8 * 8)
        x = F.relu(self.fc1(x))
        value = torch.tanh(self.fc2(x))  # Value between -1 and 1
        policy = F.softmax(self.fc3(x), dim=1)  # Probability distribution over moves
        return value, policy

# Convert chess board to neural network input
def board_to_input(board):
    # 12 planes: 6 piece types x 2 colors
    piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
    colors = [chess.WHITE, chess.BLACK]
    
    # Initialize input as 12x8x8 tensor
    input_tensor = np.zeros((12, 8, 8), dtype=np.float32)
    
    # Fill the tensor
    for i in range(64):
        piece = board.piece_at(i)
        if piece:
            piece_type = piece.piece_type - 1  # 0-5
            color = 0 if piece.color == chess.WHITE else 1
            channel = piece_type + 6 * color
            rank = i // 8
            file = i % 8
            input_tensor[channel, rank, file] = 1.0
            
    return torch.FloatTensor(input_tensor).unsqueeze(0)  # Add batch dimension

# Deep Q-Learning Agent
class ChessAgent:
    def __init__(self, epsilon=0.9, gamma=0.95):
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.model = ChessNN()
        self.target_model = ChessNN()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.batch_size = 32
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, board, training=True):
        # Epsilon-greedy action selection
        if training and np.random.rand() <= self.epsilon:
            # Random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
        
        # Get model prediction
        state = board_to_input(board)
        value, policy = self.model(state)
        
        # Convert policy to move probabilities
        move_probs = policy.detach().numpy()[0]
        
        # Get legal moves
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
            
        # Find the legal move with highest probability
        best_move = None
        best_prob = -1
        
        for move in legal_moves:
            # Convert move to index in policy output
            from_square = move.from_square
            to_square = move.to_square
            move_idx = from_square * 64 + to_square
            prob = move_probs[move_idx]
            
            if prob > best_prob:
                best_prob = prob
                best_move = move
                
        return best_move
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            state = board_to_input(state)
            next_state = board_to_input(next_state)
            
            if not done:
                next_value, _ = self.target_model(next_state)
                target = reward + self.gamma * next_value.item()
            else:
                target = reward
                
            current_value, policy = self.model(state)
            
            # Calculate loss
            value_loss = F.mse_loss(current_value, torch.tensor([[target]]))
            
            # For policy loss, we would need the action probabilities
            # This is simplified here
            from_square = action.from_square
            to_square = action.to_square
            move_idx = from_square * 64 + to_square
            
            # Create target policy
            target_policy = torch.zeros_like(policy)
            target_policy[0, move_idx] = 1.0
            
            policy_loss = F.cross_entropy(policy, target_policy)
            
            # Combined loss
            loss = value_loss + policy_loss
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        # Reduce exploration rate over time
        if self.epsilon > 0.1:
            self.epsilon *= 0.999

# Chess Game class with pygame rendering
class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((BOARD_WIDTH, BOARD_HEIGHT))
        self.clock = pygame.time.Clock()
        self.board = chess.Board()
        self.selected_square = None
        self.pieces = load_pieces()
        self.agent = ChessAgent()
        self.training_mode = False
        self.last_move = None
        
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else (100, 100, 100)
                pygame.draw.rect(self.screen, color, 
                                (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                
                # Highlight selected square
                if self.selected_square == row * 8 + col:
                    pygame.draw.rect(self.screen, HIGHLIGHT, 
                                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 4)
                                    
                # Highlight last move
                if self.last_move and (self.last_move.from_square == row * 8 + col or 
                                      self.last_move.to_square == row * 8 + col):
                    pygame.draw.rect(self.screen, LAST_MOVE, 
                                    (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
                
                # Draw pieces
                piece = self.board.piece_at(row * 8 + col)
                if piece:
                    # Use the standard symbol for internal representation
                    standard_symbol = piece.symbol()
                    self.screen.blit(self.pieces[standard_symbol], 
                                     (col * SQUARE_SIZE, row * SQUARE_SIZE))
    
    def handle_click(self, pos):
        col = pos[0] // SQUARE_SIZE
        row = pos[1] // SQUARE_SIZE
        square = row * 8 + col
        
        # If no square is selected, select this one
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
        else:
            # If a square is already selected, try to make a move
            try:
                move = chess.Move(self.selected_square, square)
                # Check for promotion
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                    if (square // 8 == 0 and self.board.turn == chess.BLACK) or \
                       (square // 8 == 7 and self.board.turn == chess.WHITE):
                        move.promotion = chess.QUEEN  # Always promote to queen for simplicity
                
                if move in self.board.legal_moves:
                    self.make_move(move)
                    self.selected_square = None
                else:
                    self.selected_square = square
            except:
                self.selected_square = square
    
    def make_move(self, move):
        # Store current state for RL training
        old_state = chess.Board(self.board.fen())
        
        # Make the move
        self.board.push(move)
        self.last_move = move
        
        # Get reward
        reward = self.get_reward()
        
        # Store experience for RL training
        if self.training_mode:
            done = self.board.is_game_over()
            self.agent.remember(old_state, move, reward, self.board, done)
            self.agent.replay()
            
        # If game not over and in AI mode, make AI move
        if not self.board.is_game_over() and self.training_mode:
            self.make_ai_move()
    
    def make_ai_move(self):
        move = self.agent.act(self.board)
        if move:
            self.make_move(move)
    
    def get_reward(self):
        # Basic reward function
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0  # White wins if black is checkmated
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0
        
        # Material count
        white_material = 0
        black_material = 0
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has infinite value but we don't count it for material
        }
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    white_material += value
                else:
                    black_material += value
        
        material_advantage = (white_material - black_material) / 100.0  # Scale down
        
        # Return from white's perspective
        return material_advantage
    
    def train(self, num_games=100):
        self.training_mode = True
        for game in range(num_games):
            self.board.reset()
            done = False
            moves_count = 0
            
            while not done and moves_count < 200:  # Add move limit to prevent infinite games
                moves_count += 1
                move = self.agent.act(self.board)
                if not move:
                    print(f"No legal moves found. Board state: {self.board.fen()}")
                    break
                    
                old_state = chess.Board(self.board.fen())
                self.board.push(move)
                reward = self.get_reward()
                done = self.board.is_game_over()
                
                self.agent.remember(old_state, move, reward, self.board, done)
                
                # Only replay periodically instead of after every move
                if moves_count % 10 == 0:
                    self.agent.replay()
                
            # Final replay at end of game
            self.agent.replay()
            
            print(f"Game {game+1}/{num_games} completed with {moves_count} moves")
            
            # Save model periodically
            if (game + 1) % 10 == 0:
                torch.save(self.agent.model.state_dict(), f"chess_model_game_{game+1}.pth")
        
        self.training_mode = False
    
    def load_model(self, path):
        if os.path.exists(path):
            self.agent.model.load_state_dict(torch.load(path))
            self.agent.update_target_model()
            print(f"Model loaded from {path}")
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    self.handle_click(pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:  # T key to toggle training mode
                        self.training_mode = not self.training_mode
                        print(f"Training mode: {'On' if self.training_mode else 'Off'}")
                    elif event.key == pygame.K_a:  # A key to make AI move
                        self.make_ai_move()
                    elif event.key == pygame.K_r:  # R key to reset board
                        self.board.reset()
                        self.selected_square = None
                        self.last_move = None
                    elif event.key == pygame.K_s:  # S key to start training
                        print("Starting training...")
                        self.train(100)  # Train for 100 games
                    elif event.key == pygame.K_l:  # L key to load model
                        self.load_model("chess_model.pth")
            
            # Draw everything
            self.screen.fill(BLACK)
            self.draw_board()
            
            # Show game status
            font = pygame.font.Font(None, 24)
            status = "Game in progress"
            if self.board.is_checkmate():
                status = "Checkmate!"
            elif self.board.is_stalemate():
                status = "Stalemate!"
            elif self.board.is_check():
                status = "Check!"
                
            text = font.render(status, True, WHITE)
            self.screen.blit(text, (10, 10))
            
            # Display current piece symbols legend
            legend_text = "White: p1,r1,n1,b1,q1,k1 | Black: p,r,n,b,q,k"
            legend = font.render(legend_text, True, WHITE)
            self.screen.blit(legend, (BOARD_WIDTH - 350, BOARD_HEIGHT - 30))
            
            mode = font.render(f"Mode: {'Training' if self.training_mode else 'Play'}", True, WHITE)
            self.screen.blit(mode, (BOARD_WIDTH - 120, 10))
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()

# Main function
def main():
    game = ChessGame()
    game.run()

if __name__ == "__main__":
    main()