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
import threading
import time
import datetime

# Constants
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 640
BOARD_WIDTH = int(WINDOW_WIDTH * 0.7)  # 70% của màn hình
BOARD_HEIGHT = WINDOW_HEIGHT
INFO_WIDTH = int(WINDOW_WIDTH * 0.3)  # 30% của màn hình
INFO_HEIGHT = WINDOW_HEIGHT
SQUARE_SIZE = BOARD_WIDTH // 8
FPS = 30
GAME_TIME = 90 * 60  # 90 phút = 5400 giây

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT = (100, 255, 100)
LAST_MOVE = (255, 255, 0)
BROWN = (139, 69, 19)
LIGHT_BROWN = (205, 133, 63)
DARK_BROWN = (101, 67, 33)
LIGHT_SQUARE = (240, 217, 181)  # Màu ô sáng
DARK_SQUARE = (181, 136, 99)  # Màu ô tối

# Trạng thái trò chơi
STATE_MENU = 0
STATE_HUMAN_VS_HUMAN = 1
STATE_HUMAN_VS_AI = 2
STATE_TRAINING = 3  # Thêm trạng thái training

# Initialize pygame
pygame.init()
pygame.display.set_caption("Chess Game")

# Tải font
def load_font(size=24):
    try:
        # Cố gắng tải font hệ thống đầu tiên
        available_fonts = pygame.font.get_fonts()
        preferred_fonts = ['arial', 'timesnewroman', 'verdana', 'dejavusans', 'calibri', 'segoeui']
        
        # Tìm font hỗ trợ tiếng Việt trong danh sách ưu tiên
        found_font = None
        for font in preferred_fonts:
            if font.lower() in available_fonts:
                found_font = font
                break
        
        if found_font:
            return pygame.font.SysFont(found_font, size)
        else:
            # Nếu không tìm thấy font ưu tiên, sử dụng font mặc định
            return pygame.font.SysFont(pygame.font.get_default_font(), size)
    except:
        # Fallback nếu có lỗi
        return pygame.font.Font(None, size)

# Loading chess piece images
def load_pieces():
    pieces = {}
    # Black pieces keep original symbols
    black_pieces = ['p', 'r', 'n', 'b', 'q', 'k']
    # White pieces use lowercase with '1' suffix
    white_pieces = ['p1', 'r1', 'n1', 'b1', 'q1', 'k1']
    
    # Load black pieces
    for piece in black_pieces:
        try:
            pieces[piece] = pygame.transform.scale(
                pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
            )
        except pygame.error as e:
            print(f"Không thể tải hình ảnh {piece}.png: {e}")
            # Tạo hình ảnh tạm thời với màu để thay thế
            surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            surf.fill((200, 0, 0))  # Màu đỏ
            pieces[piece] = surf
    
    # Load white pieces with modified names
    # But map them to standard chess notation for internal representation
    standard_white = ['P', 'R', 'N', 'B', 'Q', 'K']
    for i, piece in enumerate(white_pieces):
        try:
            pieces[standard_white[i]] = pygame.transform.scale(
                pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
            )
        except pygame.error as e:
            print(f"Không thể tải hình ảnh {piece}.png: {e}")
            # Tạo hình ảnh tạm thời với màu để thay thế
            surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
            surf.fill((0, 0, 200))  # Màu xanh
            pieces[standard_white[i]] = surf
    
    return pieces

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
    try:
        # 12 planes: 6 piece types x 2 colors
        piece_types = [chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN, chess.KING]
        
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
    except Exception as e:
        print(f"Lỗi trong board_to_input: {e}")
        # Trả về tensor rỗng nếu có lỗi
        return torch.zeros((1, 12, 8, 8), dtype=torch.float32)

# ------------------- Thêm các AI Agent mới -------------------

class RandomAgent:
    """Agent đưa ra nước đi ngẫu nhiên từ các nước đi hợp lệ"""
    
    def __init__(self):
        self.name = "Random Agent"
    
    def act(self, board, training=False):
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return random.choice(legal_moves)
        return None

class MinimaxAgent:
    """Agent sử dụng thuật toán Minimax với độ sâu hạn chế"""
    
    def __init__(self, depth=2):
        self.name = f"Minimax (depth={depth})"
        self.depth = depth
        # Giá trị của các quân cờ
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Bảng giá trị vị trí cho quân tốt (từ góc nhìn của quân trắng)
        self.pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        # Bảng giá trị vị trí cho các quân khác (đơn giản)
        self.knight_table = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]
    
    def evaluate_board(self, board):
        """Đánh giá trạng thái bàn cờ"""
        if board.is_checkmate():
            # Nếu trắng bị chiếu bí, trả về -10000, còn đen bị chiếu bí, trả về 10000
            return -10000 if board.turn == chess.WHITE else 10000
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0  # Hòa cờ
        
        total = 0
        
        # Tính toán giá trị quân cờ
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = self.piece_values[piece.piece_type]
                
                # Thêm giá trị vị trí cho quân tốt và quân mã
                if piece.piece_type == chess.PAWN:
                    # Lấy chỉ số trong bảng vị trí dựa trên màu quân
                    position_index = square if piece.color == chess.WHITE else 63 - square
                    value += self.pawn_table[position_index]
                elif piece.piece_type == chess.KNIGHT:
                    position_index = square if piece.color == chess.WHITE else 63 - square
                    value += self.knight_table[position_index]
                
                # Cộng điểm cho quân trắng, trừ điểm cho quân đen
                if piece.color == chess.WHITE:
                    total += value
                else:
                    total -= value
        
        return total
    
    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Thuật toán minimax với cắt tỉa alpha-beta"""
        if depth == 0 or board.is_game_over():
            return self.evaluate_board(board)
        
        if maximizing_player:
            max_eval = -float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def act(self, board, training=False):
        """Trả về nước đi tốt nhất dựa trên minimax"""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        best_move = legal_moves[0]
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')
        
        for move in legal_moves:
            board.push(move)
            
            # Đánh giá nước đi
            if board.turn == chess.WHITE:  # Sau khi đi, đến lượt đối phương (đen)
                value = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), False)
                if value > best_value:
                    best_value = value
                    best_move = move
            else:  # Sau khi đi, đến lượt đối phương (trắng)
                value = self.minimax(board, self.depth - 1, -float('inf'), float('inf'), True)
                if value < best_value:
                    best_value = value
                    best_move = move
            
            board.pop()
        
        return best_move

class StockfishAgent:
    """Agent sử dụng Stockfish (nếu có)"""
    
    def __init__(self, depth=1, time_limit=0.1):
        self.name = "Stockfish"
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        self.available = self._initialize_engine()
    
    def _initialize_engine(self):
        try:
            # Tìm đường dẫn đến Stockfish
            stockfish_path = self._find_stockfish()
            if stockfish_path:
                # Cố gắng import module chess.engine
                try:
                    import chess.engine
                    self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                    return True
                except ImportError:
                    print("Không thể import module chess.engine. Cần cài đặt python-chess version >= 0.23.0")
                    return False
                except Exception as e:
                    print(f"Lỗi khi khởi tạo Stockfish engine: {e}")
                    return False
            else:
                print("Không tìm thấy Stockfish. StockfishAgent không khả dụng.")
                return False
        except Exception as e:
            print(f"Lỗi khi khởi tạo Stockfish: {e}")
            return False
    
    def _find_stockfish(self):
        # Tìm kiếm Stockfish trong các vị trí thông dụng
        possible_paths = [
            "stockfish",                       # Trong PATH
            "./stockfish",                     # Thư mục hiện tại
            "./engines/stockfish",             # Thư mục engines 
            "engines/stockfish",               # Thư mục engines không có ./
            "stockfish.exe",                   # Windows
            "./stockfish.exe",                 # Windows trong thư mục hiện tại
            "./engines/stockfish.exe",         # Windows trong thư mục engines 
        ]
        
        for path in possible_paths:
            try:
                # Thử mở để xem có tồn tại không
                if os.path.exists(path):
                    return path
            except:
                pass
        return None
    
    def act(self, board, training=False):
        """Sử dụng Stockfish để tìm nước đi tốt nhất"""
        if not self.available or self.engine is None:
            # Nếu Stockfish không khả dụng, sử dụng nước đi ngẫu nhiên
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
        
        try:
            # Import lại chess.engine để đảm bảo có thể sử dụng
            import chess.engine
            
            # Sử dụng thời gian ngắn hoặc độ sâu thấp khi training
            if training:
                result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
            else:
                result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
            return result.move
        except Exception as e:
            print(f"Lỗi khi sử dụng Stockfish: {e}")
            # Fallback to random move
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
    
    def __del__(self):
        """Đảm bảo đóng engine khi đối tượng bị hủy"""
        if hasattr(self, 'available') and self.available and self.engine:
            try:
                self.engine.quit()
            except:
                pass

# ------------------- Deep Q-Learning Agent -------------------

class ChessAgent:
    def __init__(self, epsilon=0.1, gamma=0.95):
        self.name = "Deep RL Agent"
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma  # Discount factor
        self.model = ChessNN()
        self.target_model = ChessNN()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.batch_size = 32

        # Thử tải mô hình nếu có
        self.try_load_model()
        
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def act(self, board, training=False):
        try:
            # Epsilon-greedy action selection (chỉ khi training)
            if training and np.random.rand() <= self.epsilon:
                # Random move
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    return random.choice(legal_moves)
                return None
            
            # Get model prediction
            state = board_to_input(board)
            with torch.no_grad():  # Ngăn việc tính toán gradient
                value, policy = self.model(state)
            
            # Convert policy to move probabilities
            move_probs = policy.cpu().numpy()[0]
            
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
                
                # Kiểm tra giới hạn index
                if move_idx < len(move_probs):
                    prob = move_probs[move_idx]
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_move = move
            
            # Nếu không tìm thấy best_move, chọn ngẫu nhiên
            if best_move is None and legal_moves:
                best_move = random.choice(legal_moves)
                
            return best_move
        except Exception as e:
            print(f"Lỗi trong act(): {e}")
            # Trả về một nước đi hợp lệ ngẫu nhiên nếu có lỗi
            legal_moves = list(board.legal_moves)
            if legal_moves:
                return random.choice(legal_moves)
            return None
            
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self):
        try:
            if len(self.memory) < self.batch_size:
                return
                
            minibatch = random.sample(self.memory, self.batch_size)
            
            for state, action, reward, next_state, done in minibatch:
                state_input = board_to_input(state)
                next_state_input = board_to_input(next_state)
                
                # Đặt target cho giá trị
                if not done:
                    with torch.no_grad():
                        next_value, _ = self.target_model(next_state_input)
                    target = reward + self.gamma * next_value.item()
                else:
                    target = reward
                    
                current_value, policy = self.model(state_input)
                
                # Calculate loss
                value_loss = F.mse_loss(current_value, torch.tensor([[target]], dtype=torch.float32))
                
                # For policy loss, convert action to index
                try:
                    from_square = action.from_square
                    to_square = action.to_square
                    move_idx = from_square * 64 + to_square
                    
                    # Kiểm tra giới hạn index
                    if move_idx >= policy.size(1):
                        move_idx = 0
                    
                    # Create target policy
                    target_policy = torch.zeros_like(policy)
                    target_policy[0, move_idx] = 1.0
                    
                    policy_loss = F.cross_entropy(policy, target_policy)
                    
                    # Combined loss
                    loss = value_loss + policy_loss
                except Exception as e:
                    print(f"Lỗi tính policy loss: {e}")
                    loss = value_loss  # Chỉ dùng value loss nếu có lỗi
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Reduce exploration rate over time
            if self.epsilon > 0.1:
                self.epsilon *= 0.999
        except Exception as e:
            print(f"Lỗi trong replay(): {e}")
    
    def try_load_model(self):
        """Thử tải mô hình mới nhất nếu có"""
        try:
            # Nếu không có đường dẫn cụ thể, tìm model mới nhất
            model_files = [f for f in os.listdir('.') if f.startswith('chess_model_game_') and f.endswith('.pth')]
            
            if not model_files:
                # Kiểm tra xem có file chess_model.pth không
                if os.path.exists("chess_model.pth"):
                    path = "chess_model.pth"
                    print(f"Tìm thấy file model mặc định: {path}")
                    self.model.load_state_dict(torch.load(path))
                    self.update_target_model()
                    print("Đã tải model thành công!")
                    return True
                else:
                    print("Không tìm thấy file model nào.")
                    return False
            else:
                # Sắp xếp theo số ván đấu (lấy số từ tên file)
                try:
                    # Cách an toàn để lấy số từ tên file
                    def extract_game_number(filename):
                        try:
                            # Tách phần giữa 'chess_model_game_' và '.pth'
                            parts = filename.replace('chess_model_game_', '').replace('.pth', '')
                            return int(parts)
                        except ValueError:
                            # Nếu không thể chuyển đổi thành số, trả về -1
                            return -1
                    
                    model_files.sort(key=extract_game_number)
                    path = model_files[-1]  # Lấy file mới nhất
                    print(f"Tự động chọn file model mới nhất: {path}")
                    self.model.load_state_dict(torch.load(path))
                    self.update_target_model()
                    print("Đã tải model thành công!")
                    return True
                except Exception as e:
                    print(f"Lỗi khi tải model: {e}")
                    return False
        except Exception as e:
            print(f"Lỗi khi tìm model: {e}")
            return False
            
    def load_model(self, path=None):
        """Tải mô hình từ đường dẫn hoặc tìm mô hình mới nhất"""
        try:
            # Nếu không có đường dẫn cụ thể, tìm model mới nhất
            if not path:
                model_files = [f for f in os.listdir('.') if f.startswith('chess_model_game_') and f.endswith('.pth')]
                
                if not model_files:
                    # Kiểm tra xem có file chess_model.pth không
                    if os.path.exists("chess_model.pth"):
                        path = "chess_model.pth"
                        print(f"Tìm thấy file model mặc định: {path}")
                    else:
                        print("Không tìm thấy file model nào. Hãy train mô hình trước.")
                        return False
                else:
                    # Sắp xếp theo số ván đấu (lấy số từ tên file)
                    try:
                        # Cách an toàn để lấy số từ tên file
                        def extract_game_number(filename):
                            try:
                                # Tách phần giữa 'chess_model_game_' và '.pth'
                                parts = filename.replace('chess_model_game_', '').replace('.pth', '')
                                return int(parts)
                            except ValueError:
                                # Nếu không thể chuyển đổi thành số, trả về -1
                                return -1
                        
                        model_files.sort(key=extract_game_number)
                        path = model_files[-1]  # Lấy file mới nhất
                        print(f"Tự động chọn file model mới nhất: {path}")
                    except Exception as e:
                        # Nếu có lỗi trong quá trình sắp xếp, sử dụng file mặc định
                        if os.path.exists("chess_model.pth"):
                            path = "chess_model.pth"
                            print(f"Lỗi khi sắp xếp files: {e}")
                            print(f"Sử dụng file model mặc định: {path}")
                        else:
                            print(f"Lỗi khi sắp xếp files: {e}")
                            return False
            
            if os.path.exists(path):
                self.model.load_state_dict(torch.load(path))
                self.update_target_model()
                print(f"Đã tải model từ {path}")
                return True
            else:
                print(f"Không tìm thấy file model tại {path}")
                return False
        except Exception as e:
            print(f"Lỗi khi tải model: {e}")
            return False

# ------------------- Lớp Menu chính -------------------

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        self.font_title = load_font(64)
        self.font = load_font(36)
        self.selected_option = 0
        self.options = ["Người - Người", "Người - Máy", "Huấn luyện AI"]
        
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.options)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.options)
            elif event.key == pygame.K_RETURN:
                if self.selected_option == 0:
                    return STATE_HUMAN_VS_HUMAN
                elif self.selected_option == 1:
                    return STATE_HUMAN_VS_AI
                else:
                    return STATE_TRAINING
        return STATE_MENU
        
    def draw(self):
        # Vẽ nền
        self.screen.fill(DARK_BROWN)
        
        # Vẽ tiêu đề
        title = self.font_title.render("Chess Game", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Vẽ các lựa chọn
        for i, option in enumerate(self.options):
            color = LIGHT_BROWN if i == self.selected_option else WHITE
            text = self.font.render(option, True, color)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, 300 + i * 80))
            
            # Vẽ hình chữ nhật nổi bật cho lựa chọn hiện tại
            if i == self.selected_option:
                pygame.draw.rect(self.screen, LIGHT_BROWN, 
                                 (text_rect.left - 20, text_rect.top - 10, 
                                  text_rect.width + 40, text_rect.height + 20), 2)
                
            self.screen.blit(text, text_rect)
                
        # Vẽ hướng dẫn
        guide = self.font.render("Su dung phim mui ten de chon, Enter de xac nhan", True, WHITE)
        guide_rect = guide.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
        self.screen.blit(guide, guide_rect)

# ------------------- Menu chọn đối thủ training -------------------

class TrainingMenu:
    """Menu lựa chọn đối thủ để train AI"""
    def __init__(self, screen, agents):
        self.screen = screen
        self.font_title = load_font(36)
        self.font = load_font(24)
        self.selected_option = 0
        self.agents = agents  # Danh sách các agent khả dụng
        
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                self.selected_option = (self.selected_option - 1) % len(self.agents)
            elif event.key == pygame.K_DOWN:
                self.selected_option = (self.selected_option + 1) % len(self.agents)
            elif event.key == pygame.K_RETURN:
                return self.selected_option
            elif event.key == pygame.K_ESCAPE:
                return -1  # Trở về menu chính
        return None
        
    def draw(self):
        # Vẽ nền
        self.screen.fill(DARK_BROWN)
        
        # Vẽ tiêu đề
        title = self.font_title.render("Chon doi thu de training", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH // 2, 100))
        self.screen.blit(title, title_rect)
        
        # Vẽ các lựa chọn
        y_pos = 180
        for i, agent in enumerate(self.agents):
            color = LIGHT_BROWN if i == self.selected_option else WHITE
            text = self.font.render(agent.name, True, color)
            text_rect = text.get_rect(center=(WINDOW_WIDTH // 2, y_pos))
            
            # Vẽ hình chữ nhật nổi bật cho lựa chọn hiện tại
            if i == self.selected_option:
                pygame.draw.rect(self.screen, LIGHT_BROWN, 
                                (text_rect.left - 20, text_rect.top - 10, 
                                text_rect.width + 40, text_rect.height + 20), 2)
                
            self.screen.blit(text, text_rect)
            y_pos += 60
                
        # Vẽ hướng dẫn
        guide = self.font.render("Su dung phim mui ten de chon, Enter de xac nhan", True, WHITE)
        guide_rect = guide.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 100))
        self.screen.blit(guide, guide_rect)
        
        esc_guide = self.font.render("ESC: Quay lai", True, WHITE)
        esc_rect = esc_guide.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT - 60))
        self.screen.blit(esc_guide, esc_rect)

# ------------------- Chess Game class -------------------

class ChessGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.state = STATE_MENU
        
        # Khởi tạo các AI agent
        self.agent = ChessAgent()  # Deep RL Agent chính
        self.training_agents = [
            self.agent,                # Self-play (đánh với chính mình)
            RandomAgent(),             # Agent ngẫu nhiên
            MinimaxAgent(depth=2),     # Agent Minimax độ sâu 2
            StockfishAgent(depth=1)    # Stockfish nếu có
        ]
        
        # Lọc ra các agent khả dụng (loại bỏ StockfishAgent nếu không tìm thấy Stockfish)
        self.available_agents = [agent for agent in self.training_agents if not hasattr(agent, 'available') or agent.available]
        
        # Khởi tạo các menu
        self.menu = MainMenu(self.screen)
        self.training_menu = TrainingMenu(self.screen, self.available_agents)
        
        # Đối thủ training hiện tại
        self.opponent_agent = self.agent  # Mặc định là self-play
        
        # Game elements
        self.board = chess.Board()
        self.selected_square = None
        self.pieces = load_pieces()
        self.last_move = None
        
        # Timer
        self.white_time = GAME_TIME
        self.black_time = GAME_TIME
        self.last_move_time = time.time()
        
        # Training status
        self.is_training = False
        self.training_thread = None
        self.training_status = "San sang"
        self.games_completed = 0
        self.showing_training_menu = False
        
        # Fonts
        self.font_large = load_font(32)
        self.font_medium = load_font(24)
        self.font_small = load_font(18)
        
    def reset_game(self):
        self.board = chess.Board()
        self.selected_square = None
        self.last_move = None
        self.white_time = GAME_TIME
        self.black_time = GAME_TIME
        self.last_move_time = time.time()
        
    def draw_board(self):
        # Vẽ bàn cờ
        for row in range(8):
            for col in range(8):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
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
        
        # Vẽ nhãn cột và hàng
        for i in range(8):
            # Số hàng (8->1)
            text = self.font_small.render(str(8-i), True, BLACK if i % 2 == 0 else WHITE)
            self.screen.blit(text, (5, i * SQUARE_SIZE + 5))
            
            # Chữ cột (a->h)
            text = self.font_small.render(chr(97+i), True, BLACK if (i+7) % 2 == 0 else WHITE)
            self.screen.blit(text, (i * SQUARE_SIZE + SQUARE_SIZE - 15, BOARD_HEIGHT - 20))
    
    def draw_info_panel(self):
        # Vẽ panel thông tin bên phải
        pygame.draw.rect(self.screen, BROWN, (BOARD_WIDTH, 0, INFO_WIDTH, INFO_HEIGHT))
        
        # Vẽ đường phân cách
        pygame.draw.line(self.screen, WHITE, (BOARD_WIDTH, 0), (BOARD_WIDTH, BOARD_HEIGHT), 3)
        
        # Hiển thị thời gian
        white_min = int(self.white_time // 60)
        white_sec = int(self.white_time % 60)
        black_min = int(self.black_time // 60)
        black_sec = int(self.black_time % 60)
        
        # Sửa lại nhãn: Trắng hiển thị cho quân trắng, Đen hiển thị cho quân đen
        white_time_text = f"Trang: {white_min:02d}:{white_sec:02d}"
        black_time_text = f"Den: {black_min:02d}:{black_sec:02d}"
        
        # Đổi màu cho đồng hồ của người đang đi
        white_color = HIGHLIGHT if self.board.turn == chess.WHITE else WHITE
        black_color = HIGHLIGHT if self.board.turn == chess.BLACK else WHITE
        
        white_time_surface = self.font_medium.render(white_time_text, True, white_color)
        black_time_surface = self.font_medium.render(black_time_text, True, black_color)
        
        self.screen.blit(white_time_surface, (BOARD_WIDTH + 20, 30))
        self.screen.blit(black_time_surface, (BOARD_WIDTH + 20, 70))
        
        # Vẽ lượt đi hiện tại
        turn_text = "Luot: " + ("Trang" if self.board.turn == chess.WHITE else "Den")
        turn_surface = self.font_medium.render(turn_text, True, WHITE)
        self.screen.blit(turn_surface, (BOARD_WIDTH + 20, 120))
        
        # Game status
        status = ""
        if self.board.is_checkmate():
            status = "Chieu bi!"
            winner = "Trang" if not self.board.turn else "Den"
            status += f" {winner} thang!"
        elif self.board.is_stalemate():
            status = "Hoa co - be tac!"
        elif self.board.is_insufficient_material():
            status = "Hoa co - thieu quan!"
        elif self.board.is_check():
            status = "Chieu tuong!"
        
        if status:
            status_surface = self.font_medium.render(status, True, WHITE)
            self.screen.blit(status_surface, (BOARD_WIDTH + 10, 170))
        
        # Chế độ chơi
        mode = "Che do: "
        if self.state == STATE_HUMAN_VS_HUMAN:
            mode += "Nguoi-Nguoi"
        elif self.state == STATE_HUMAN_VS_AI:
            mode += "Nguoi-May"
        else:
            mode += "Training"
        
        mode_surface = self.font_small.render(mode, True, WHITE)
        self.screen.blit(mode_surface, (BOARD_WIDTH + 10, 220))
        
        # Hiển thị nước đi gần nhất
        if self.last_move:
            move_text = "Nuoc di gan nhat: "
            move_uci = self.last_move.uci()
            from_sq = move_uci[:2]
            to_sq = move_uci[2:4]
            move_text += f"{from_sq}->{to_sq}"
            
            move_surface = self.font_small.render(move_text, True, WHITE)
            self.screen.blit(move_surface, (BOARD_WIDTH + 10, 250))
        
        # Hiển thị trạng thái training nếu đang training
        if self.state == STATE_TRAINING:
            if self.is_training:
                train_text = f"Training: {self.games_completed}"
                train_surface = self.font_small.render(train_text, True, LIGHT_BROWN)
                self.screen.blit(train_surface, (BOARD_WIDTH + 10, 280))
                
                # Hiển thị đối thủ đang training cùng
                opponent_text = f"Doi thu: {self.opponent_agent.name}"
                opponent_surface = self.font_small.render(opponent_text, True, WHITE)
                self.screen.blit(opponent_surface, (BOARD_WIDTH + 10, 310))
                
                status_text = f"{self.training_status}"
                status_surface = self.font_small.render(status_text, True, WHITE)
                self.screen.blit(status_surface, (BOARD_WIDTH + 10, 340))
            else:
                train_text = "Ban co the:"
                train_surface = self.font_small.render(train_text, True, WHITE)
                self.screen.blit(train_surface, (BOARD_WIDTH + 10, 280))
                
                options = [
                    "T: Chon doi thu",
                    "S: Bat dau training",
                    "L: Tai model",
                    "X: Dung training",
                    "ESC: Quay lai menu"
                ]
                
                y_pos = 310
                for option in options:
                    opt_surface = self.font_small.render(option, True, LIGHT_BROWN)
                    self.screen.blit(opt_surface, (BOARD_WIDTH + 10, y_pos))
                    y_pos += 25
        
        # Hướng dẫn
        if self.state != STATE_TRAINING:
            instructions = [
                "Huong dan:",
                "R: Choi lai",
                "ESC: Ve menu",
                "Click: Chon quan/nuoc di"
            ]
            
            y_pos = 300
            for instruction in instructions:
                inst_surface = self.font_small.render(instruction, True, WHITE)
                self.screen.blit(inst_surface, (BOARD_WIDTH + 10, y_pos))
                y_pos += 30

    def update_timer(self):
        if self.state == STATE_TRAINING or self.board.is_game_over():
            return
            
        current_time = time.time()
        if self.last_move_time is not None:
            elapsed = current_time - self.last_move_time
            
            # Người đi lượt hiện tại sẽ bị trừ thời gian
            if self.board.turn == chess.WHITE:
                self.white_time = max(0, self.white_time - elapsed)
            else:
                self.black_time = max(0, self.black_time - elapsed)
        
        self.last_move_time = current_time
    
    def handle_click(self, pos):
        # Chỉ xử lý click nếu nằm trong bàn cờ
        if pos[0] >= BOARD_WIDTH:
            return
            
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
                if self.board.piece_at(self.selected_square) and self.board.piece_at(self.selected_square).piece_type == chess.PAWN:
                    if (square // 8 == 0 and self.board.turn == chess.BLACK) or \
                       (square // 8 == 7 and self.board.turn == chess.WHITE):
                        move.promotion = chess.QUEEN  # Always promote to queen for simplicity
                
                if move in self.board.legal_moves:
                    self.make_move(move)
                    self.selected_square = None
                else:
                    self.selected_square = square
            except Exception as e:
                print(f"Lỗi khi thực hiện nước đi: {e}")
                self.selected_square = square
    
    def make_move(self, move):
        try:
            # Cập nhật thời gian trước khi đi nước mới
            self.update_timer()
            
            # Make the move
            self.board.push(move)
            self.last_move = move
            
            # Nếu đang ở chế độ người-máy và đến lượt máy
            if self.state == STATE_HUMAN_VS_AI and self.board.turn == chess.BLACK and not self.board.is_game_over():
                # Thêm chút độ trễ để người chơi thấy được nước đi của mình trước
                pygame.time.delay(500)
                self.make_ai_move()
                
        except Exception as e:
            print(f"Lỗi trong make_move: {e}")
    
    def make_ai_move(self):
        try:
            # Cập nhật thời gian trước khi máy đi
            self.update_timer()
            
            # Lấy nước đi tốt nhất từ AI
            move = self.agent.act(self.board)
            if move:
                # Make the move
                self.board.push(move)
                self.last_move = move
                
                # Cập nhật thời gian sau khi máy đi
                self.update_timer()
        except Exception as e:
            print(f"Lỗi trong make_ai_move: {e}")
    
    def _get_reward_for_board(self, board):
        """Tính reward cho một bàn cờ cụ thể (sử dụng trong thread training)"""
        try:
            if board.is_checkmate():
                return 1.0 if board.turn == chess.BLACK else -1.0
            elif board.is_stalemate() or board.is_insufficient_material():
                return 0.0
            
            white_material = 0
            black_material = 0
            piece_values = {
                chess.PAWN: 1,
                chess.KNIGHT: 3,
                chess.BISHOP: 3,
                chess.ROOK: 5,
                chess.QUEEN: 9,
                chess.KING: 0
            }
            
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece:
                    value = piece_values[piece.piece_type]
                    if piece.color == chess.WHITE:
                        white_material += value
                    else:
                        black_material += value
            
            return (white_material - black_material) / 100.0
        except Exception as e:
            print(f"Lỗi trong _get_reward_for_board: {e}")
            return 0.0
    
    def train_worker(self, num_games=100):
        """Chạy quá trình training trong một thread riêng biệt"""
        try:
            self.is_training = True
            self.games_completed = 0
            
            for game in range(num_games):
                if not self.is_training:  # Cho phép dừng quá trình training
                    break
                    
                self.training_status = f"Dang train tran {game+1}/{num_games}"
                game_board = chess.Board()  # Dùng một bàn cờ mới để tránh xung đột
                done = False
                moves_count = 0
                max_moves = 200  # Giới hạn số lượng nước đi để tránh vòng lặp vô hạn
                
                # RL Agent luôn đi quân trắng, đối thủ đi quân đen
                while not done and moves_count < max_moves:
                    # RL Agent's turn (WHITE)
                    if game_board.turn == chess.WHITE:
                        move = self.agent.act(game_board, training=True)
                    else:
                        # Opponent's turn (BLACK)
                        move = self.opponent_agent.act(game_board, training=True)
                    
                    if not move:
                        break
                        
                    old_state = chess.Board(game_board.fen())
                    game_board.push(move)
                    reward = self._get_reward_for_board(game_board)
                    done = game_board.is_game_over()
                    
                    # Chỉ lưu kinh nghiệm cho RL Agent (quân trắng)
                    if old_state.turn == chess.WHITE:
                        self.agent.remember(old_state, move, reward, game_board, done)
                    
                    # Chỉ gọi replay() sau mỗi 10 nước đi để tăng hiệu suất
                    if moves_count % 10 == 0:
                        self.agent.replay()
                    
                    moves_count += 1
                
                self.games_completed = game + 1
                print(f"Game {game+1}/{num_games} hoàn thành sau {moves_count} nước")
                
                # Cập nhật target model sau mỗi trận đấu
                self.agent.update_target_model()
                
                # Save model periodically
                if (game + 1) % 10 == 0:
                    try:
                        model_path = f"chess_model_game_{game+1}.pth"
                        torch.save(self.agent.model.state_dict(), model_path)
                        print(f"Đã lưu mô hình tại {model_path}")
                        
                        # Lưu thêm một bản sao với tên cố định để dễ tải sau này
                        torch.save(self.agent.model.state_dict(), "chess_model.pth")
                        print("Đã lưu bản sao tại chess_model.pth")
                    except Exception as e:
                        print(f"Lỗi khi lưu mô hình: {e}")
            
            self.training_status = f"Da hoan thanh {self.games_completed}/{num_games} tran"
            self.is_training = False
        except Exception as e:
            self.training_status = f"Loi training: {str(e)}"
            self.is_training = False
            print(f"Lỗi trong train_worker: {e}")
    
    def start_training(self, num_games=100):
        """Bắt đầu quá trình training trong một thread riêng"""
        if self.is_training:
            print("Đang trong quá trình training, không thể bắt đầu mới")
            return
            
        self.training_status = "Dang chuan bi..."
        self.training_thread = threading.Thread(target=self.train_worker, args=(num_games,))
        self.training_thread.daemon = True  # Thread sẽ kết thúc khi chương trình chính kết thúc
        self.training_thread.start()
    
    def run(self):
        running = True
        
        while running:
            # Xử lý các sự kiện
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Xử lý sự kiện dựa trên trạng thái hiện tại
                if self.state == STATE_MENU:
                    new_state = self.menu.handle_event(event)
                    if new_state != STATE_MENU:
                        self.state = new_state
                        if new_state != STATE_TRAINING:
                            self.reset_game()
                            self.last_move_time = time.time()
                elif self.state == STATE_TRAINING:
                    if self.showing_training_menu:
                        # Trong menu chọn đối thủ training
                        agent_index = self.training_menu.handle_event(event)
                        if agent_index is not None:
                            if agent_index >= 0:
                                self.opponent_agent = self.available_agents[agent_index]
                                print(f"Đã chọn đối thủ training: {self.opponent_agent.name}")
                            self.showing_training_menu = False
                    else:
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.state = STATE_MENU
                                # Dừng training nếu đang chạy
                                if self.is_training:
                                    self.is_training = False
                                    self.training_status = "Đã dừng training"
                            elif event.key == pygame.K_t and not self.is_training:
                                # Hiển thị menu chọn đối thủ
                                self.showing_training_menu = True
                            elif event.key == pygame.K_s and not self.is_training:
                                print("Bắt đầu quá trình training...")
                                self.start_training(100)  # Train for 100 games
                            elif event.key == pygame.K_l and not self.is_training:
                                self.agent.load_model()
                                self.training_status = "Da tai model thanh cong"
                            elif event.key == pygame.K_x and self.is_training:
                                self.is_training = False
                                self.training_status = "Da dung training thu cong"
                                print("Đã gửi yêu cầu dừng quá trình training...")
                else:  # STATE_HUMAN_VS_HUMAN or STATE_HUMAN_VS_AI
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        self.handle_click(pos)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.state = STATE_MENU
                        elif event.key == pygame.K_r:
                            self.reset_game()
            
            # Cập nhật thời gian nếu đang trong chế độ chơi
            if self.state not in [STATE_MENU, STATE_TRAINING] and not self.board.is_game_over():
                current_time = time.time()
                if self.last_move_time is not None:
                    elapsed = current_time - self.last_move_time
                    
                    if self.board.turn == chess.WHITE:
                        self.white_time = max(0, self.white_time - elapsed)
                    else:
                        self.black_time = max(0, self.black_time - elapsed)
                
                self.last_move_time = current_time
            
            # Vẽ giao diện dựa trên trạng thái hiện tại
            if self.state == STATE_MENU:
                self.menu.draw()
            elif self.state == STATE_TRAINING and self.showing_training_menu:
                self.training_menu.draw()
            else:  # In game or training
                self.draw_board()
                self.draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(FPS)
            
        # Đảm bảo đóng các resources khi thoát
        if hasattr(self, 'training_agents'):
            for agent in self.training_agents:
                if hasattr(agent, '__del__'):
                    try:
                        agent.__del__()
                    except:
                        pass
                        
        pygame.quit()

# Main function
def main():
    try:
        game = ChessGame()
        game.run()
    except Exception as e:
        print(f"Lỗi chính: {e}")
        pygame.quit()

if __name__ == "__main__":
    main()