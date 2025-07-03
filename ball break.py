#import all the tool we are going to use 
import pygame
import math
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()
#add the game constants 
BALL_SIZE = 20
PADDLE_WIDTH = 60
PADDLE_HEIGHT = 10
PADDLE_SPEED = 5
#
class QLearningBreakout:
    #defining the parameters of the game like movements
    def __init__(self):
        self.reset()
        self.q_table = defaultdict(lambda: np.zeros(3))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.3
        self.last_wall_hit = False
        self.first_paddle_hit = True
        self.episode_reward = 0
    #starts new episode    
    def reset(self):
        self.ball = pygame.Rect(200, 200, BALL_SIZE, BALL_SIZE)
        self.ball_speed = [0, 0]
        self.paddle = pygame.Rect(180, 380, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.score = 0
        self.episode_reward = 0
        self.angle = math.radians(75)
        self.done = False
        self.last_wall_hit = False
        self.first_paddle_hit = True
        return self._get_state()
    #
    def _get_state(self):
        return (
            self.ball.x // 20,
            self.ball.y // 20,
            np.sign(self.ball_speed[0]),
            np.sign(self.ball_speed[1]), 
            self.paddle.x // 20,      
            int(self.last_wall_hit),  
            int(self.first_paddle_hit) 
        )
    #defines the movement of the paddle, the reward and the penalty also the ruke that the ball has to hit the wall
    def step(self, action):
        if action == 0 and self.paddle.left > 0:
            self.paddle.x -= PADDLE_SPEED
        elif action == 2 and self.paddle.right < 400:
            self.paddle.x += PADDLE_SPEED

        if self.ball_speed == [0, 0]:
            self.ball_speed = [3 * math.cos(self.angle), 3 * math.sin(self.angle)]
            
        self.ball.x += self.ball_speed[0]
        self.ball.y += self.ball_speed[1]
 
        reward = 0
        wall_hit = False
        
        # Wall collisions
        if self.ball.left < 0 or self.ball.right > 400:
            self.ball_speed[0] *= -1
            wall_hit = True
        if self.ball.top < 0:
            self.ball_speed[1] *= -1
            wall_hit = True

        if wall_hit:
            self.last_wall_hit = True
            
        if self.ball.colliderect(self.paddle):
            if not self.first_paddle_hit and not self.last_wall_hit:
                reward = -5 
            else:
                self.ball_speed[1] *= -1
                self.score += 1
                reward = 10
                self.ball_speed[0] *= 1.1
                self.ball_speed[1] *= 1.1
                self.paddle.width = max(20, self.paddle.width - 2)
                self.last_wall_hit = False
                self.first_paddle_hit = False 

        if self.ball.bottom > 400:
            reward = -10
            self.done = True
            
        return self._get_state(), reward, self.done
    
    def get_action(self, state):
        if random.random() < self.exploration_rate:
            return random.randint(0, 2)
        return np.argmax(self.q_table[state])
    #this is what makes the q learning agent learn by updating the toward the best course of action
    def learn(self, state, action, reward, next_state):
        best_next = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        #defines the amount of episodes or number of games 
def train_agent(episodes=1000):
    env = QLearningBreakout()
    all_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = env.get_action(state)
            next_state, reward, done = env.step(action)
            env.learn(state, action, reward, next_state)
            state = next_state
            env.episode_reward += reward
            
            if episode % 100 == 0:
                screen.fill((0, 0, 0))
                pygame.draw.ellipse(screen, (255, 0, 0), env.ball)
                pygame.draw.rect(screen, (0, 255, 0), env.paddle)
                font = pygame.font.Font(None, 36)
                texts = [
                    f"Ep: {episode}",
                    f"Score: {env.score}",
                    f"Reward: {env.episode_reward:.1f}"
                ]
                for i, text in enumerate(texts):
                    text_surface = font.render(text, True, (255, 255, 255))
                    screen.blit(text_surface, (10, 10 + i * 30))
                pygame.display.flip()
                clock.tick(60)
                
        all_rewards.append(env.episode_reward)
        
        if episode % 100 == 0:
            avg = np.mean(all_rewards[-100:]) if episode > 0 else 0
            print(f"Episode {episode}, Reward: {env.episode_reward:.1f}, Avg: {avg:.1f}")
            
    pygame.quit()
    return all_rewards

all_rewards = train_agent(episodes=1500)

#viualises the data 
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Ball break q learning progress")
plt.show()