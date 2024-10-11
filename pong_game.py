import pygame
import random
import numpy as np

class PongEnv:
    def __init__(self):
        pygame.init()

        self.game_speed = 4

        self.screen_width = 800
        self.screen_height = 600
        self.paddle_width = 10
        self.paddle_height = 100
        self.ball_size = 20
        self.paddle_speed = 6 * self.game_speed
        self.ball_speed = [5 * self.game_speed, 5 * self.game_speed]
        self.oponent_speed = 2 * self.game_speed
        self.toggle_manual_override = False
        
        # Colors
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pong - RL Environment")
        self.clock = pygame.time.Clock()

        # Initialize game objects
        self.paddle_left = pygame.Rect(20, (self.screen_height - self.paddle_height) // 2, self.paddle_width, self.paddle_height)
        self.paddle_right = pygame.Rect(self.screen_width - 30, (self.screen_height - self.paddle_height) // 2, self.paddle_width, self.paddle_height)
        self.ball = pygame.Rect(self.screen_width // 2 - self.ball_size // 2, self.screen_height // 2 - self.ball_size // 2, self.ball_size, self.ball_size)
        self.ball_speed_x = random.choice((5, -5))
        self.ball_speed_y = random.choice((5, -5))
        
        self.done = False
        self.reward = 0

    def reset(self):
        # Reset the game to the initial state
        self.paddle_left.y = (self.screen_height - self.paddle_height) // 2
        self.paddle_right.y = (self.screen_height - self.paddle_height) // 2
        self.ball.x = self.screen_width // 2 - self.ball_size // 2
        self.ball.y = self.screen_height // 2 - self.ball_size // 2
        self.ball_speed_x = random.choice((5, -5))
        self.ball_speed_y = random.choice((5, -5))
        self.done = False
        return self.get_observation()

    def get_observation(self):
        # Return the state (observation)
        return np.array([
            self.paddle_left.y,               # Player paddle y position
            self.paddle_right.y,              # Opponent paddle y position
            self.ball.x, self.ball.y,         # Ball position
            self.ball_speed_x, self.ball_speed_y  # Ball velocity
        ], dtype=np.float32)

    def step(self, action):
        # Override ai action if user is controlling the paddle
        keys = pygame.key.get_pressed()

        if self.toggle_manual_override:
            action = 2
        if keys[pygame.K_w] and self.toggle_manual_override:
            action = 0
        if keys[pygame.K_s] and self.toggle_manual_override:
            action = 1
        if keys[pygame.K_q]:
            self.toggle_manual_override = True
        if keys[pygame.K_e]:
            self.toggle_manual_override = False
        
        # Action is either 0 (up), 1 (down), or 2 (stay)
        if action == 0 and self.paddle_left.top > 0:
            self.paddle_left.y -= self.paddle_speed
        if action == 1 and self.paddle_left.bottom < self.screen_height:
            self.paddle_left.y += self.paddle_speed

        # Small penalty for moving the paddle too often in one direction
        if action == 0:
            self.reward -= 0.01
        
        if action == 1:
            self.reward -= 0.01

        # Move the ball
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y

        # Ball collision with walls
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y *= -1
        
        # Ball collision with paddles
        if self.ball.colliderect(self.paddle_left):
            self.ball_speed_x = self.check_paddle_collision(self.paddle_left, self.ball_speed_x) * -1

        if self.ball.colliderect(self.paddle_right):
            self.ball_speed_x = self.check_paddle_collision(self.paddle_right, self.ball_speed_x) * -1

        # If the ball hits the edges of the paddles, the ball will speed up

        
        # Check if ball goes out of bounds (score)
        if self.ball.left <= 0:  # AI misses the ball (negative reward)
            self.reward -= 1
            self.done = True
        elif self.ball.right >= self.screen_width:  # AI scores (positive reward)
            self.reward += 1
            self.done = True
        else:
            self.reward = 0.01  # Small positive reward for keeping the ball in play
        
        # Opponent paddle AI (simple AI that follows the ball) multiply the paddle speed by a random number to slow down the AI
        if self.ball.y < self.paddle_right.centery:
            self.paddle_right.y -= self.oponent_speed
        if self.ball.y > self.paddle_right.centery:
            self.paddle_right.y += self.oponent_speed

        return self.get_observation(), self.reward, self.done
    
    def check_paddle_collision(self, paddle, ball_speed_x):
        """
        Handles the collision between the ball and a paddle, with logic to increase ball speed
        based on how close the hit was to the top or bottom edges of the paddle.
        """
        # Get the positions of the paddle
        paddle_mid = paddle.centery
        paddle_top = paddle.top
        paddle_bottom = paddle.bottom
        ball_y = self.ball.centery  # The vertical position of the ball at the time of collision

        # Calculate the distances from the ball to the top, bottom, and center of the paddle
        distance_from_top = abs(ball_y - paddle_top)
        distance_from_bottom = abs(ball_y - paddle_bottom)
        distance_from_center = abs(ball_y - paddle_mid)

        # Define a threshold for how close the ball needs to be to the edges to count as an edge hit
        edge_threshold = paddle.height * 0.2  # 20% of the paddle height considered as edge

        # Speed up the ball if it hits near the top or bottom edges
        if distance_from_top < edge_threshold or distance_from_bottom < edge_threshold:
            print("Edge hit")
            # Ball hit near the edge of the paddle, increase speed
            ball_speed_x *= 1.3  # Increase horizontal speed by 30%
            # Adjust vertical speed based on how far from the center the ball hit
            self.ball_speed_y += (distance_from_center / paddle.height) * 10  # Scale vertical speed

        # Reflect the ball's horizontal speed (reverse direction is handled in the main loop)
        return ball_speed_x


    def render(self):
        # Render the game on the screen
        self.screen.fill(self.black)
        if self.toggle_manual_override:
            pygame.draw.rect(self.screen, (255, 0, 0), self.paddle_left)
        else:
            pygame.draw.rect(self.screen, self.white, self.paddle_left)
        pygame.draw.rect(self.screen, self.white, self.paddle_right)
        pygame.draw.ellipse(self.screen, self.white, self.ball)
        pygame.draw.aaline(self.screen, self.white, (self.screen_width // 2, 0), (self.screen_width // 2, self.screen_height))
        pygame.display.flip()
        self.clock.tick(60)
        pygame.event.pump()

    def close(self):
        pygame.quit()


if __name__ == "__main__":
    env = PongEnv()
    state = env.reset()

    for _ in range(1000):  # Play for 1000 frames
        action = random.choice([0, 1, 2])  # Random action: stay (0), up (1), or down (2)
        state, reward, done = env.step(action)
        env.render()

        if done:
            state = env.reset()

    env.close()
