import pygame
import math
import numpy as np
import random
import csv
import json
from prodict import Prodict

class Wall:
    def __init__(self, start_pos, end_pos, type='none', color = 'black', **kwargs):
        self.start_pos = start_pos
        self.end_pos = end_pos
        if color == "__random__":
            colors = ['orchid', 'white', 'palegreen', 'sienna1', 'yellow1','cyan2', 'cornsilk1']
            self.color = random.choice(colors)
        else:
            self.color = color

        self.type = type
        self.slope_x = end_pos[0] - start_pos[0]
        self.slope_y = end_pos[1] - start_pos[1]
        if self.slope_x == 0:
            self.slope = 0
            self.angle = 0
        else:
            self.slope = self.slope_y / self.slope_x
            self.angle = np.arctan(self.slope)

        self.length = math.sqrt(self.slope_x**2 + self.slope_y**2)

    def draw(self, display):
        pygame.draw.line(display, self.color, self.start_pos, self.end_pos, 3)
        return display

    def save(self,):
        return {
            'end_pos': self.end_pos,
            'start_pos': self.start_pos,
            'type': 'self.type',
            'color': self.color
        }
    
