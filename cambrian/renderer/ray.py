import math 
import numpy as np

class Ray:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.dir = (math.cos(angle), math.sin(angle))
        self.intensity = None # rgb value # optional  
        self.collision_point = None # end point # optional 

    def update(self, mx, my):
        self.x = mx
        self.y = my
    
    def checkPyGameRectCollision(self, rect, extents=[800, 800]):
        # line dist should be bigger than the longest line possible
        _t = np.sqrt(extents[0] ** 2 + extents[1]**2) + 100 
        x1 = self.x
        y1 = self.y
        x2 = self.x + _t * self.dir[0]
        y2 = self.y + _t * self.dir[1]
        # x2 = np.clip(x2, 0, extents[0])
        # y2 = np.clip(y2, 0, extents[1])

        clipped_line = rect.clipline((x1, y1), (x2, y2))
        if clipped_line: 
            start, end = clipped_line
            sx1, sy1 = start # start will always be closer
            ex1, ey2 = end
            dist = math.sqrt((x1 - sx1)**2 + (y1 - sy1)**2)
            return [sx1, sy1], dist
        else:
            return None 

    def checkCollision(self, wall):
        """
        Checks Ray-Wall Collision & output: 
        - collidePos
        - wall.color
        - wall.type
        """
        x1 = wall.start_pos[0]
        y1 = wall.start_pos[1]
        x2 = wall.end_pos[0]
        y2 = wall.end_pos[1]

        x3 = self.x
        y3 = self.y
        x4 = self.x + self.dir[0]
        y4 = self.y + self.dir[1]
    
        # Using line-line intersection formula to get intersection point of ray and wall
        # Where (x1, y1), (x2, y2) are the ray pos and (x3, y3), (x4, y4) are the wall pos
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        numerator = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
        if denominator == 0:
            return None # if no collision then the color of the wall is black
        
        t = numerator / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator

        if 1 > t > 0 and u > 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            collidePos = [x, y]
            return collidePos, wall
