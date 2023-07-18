import numpy as np
import random
import json

from wall import Wall

class Tunnel: 
    def __init__(self,
                 geometry_cfg,
                 window_size,
                 walls=None):
        """
        window_size: (x,y) tuple specifying size of window
        left_wall_color_time_period: int specifying how many pixels between color changes on left walls
            (must be less than or equal to min_segment length)
        right_wall_time_period_ratio: int specifying how much color changes on right wall w.r.t. left wall
            (e.g. if this is 2 and left_wall_color_time_period is 10 then right wall color changes every 20 pixels)
        min_seg_length: int specifying min distance before wall direction changes (e.g. tunnel turn)
        max_seg_length: int specifying max distance before wall direction changes (e.g. tunnel turn)
        tunnel_width: int specifying distance between walls in tunnel
        theta_range: the range of angles that the tunnel can tunnel can turn
        wall_colors: wall colors to alternate between (can add arbritary number of colors)
        """
        self.cfg = geometry_cfg

        self.window_size = window_size
        self.min_seg_length = geometry_cfg.min_seg_length
        self.max_seg_length = geometry_cfg.max_seg_length
        self.tunnel_width = geometry_cfg.tunnel_width
        self.theta_range = geometry_cfg.theta_range
        self.left_wall_color_time_period = geometry_cfg.left_wall_color_time_period
        self.right_wall_time_period_ratio = geometry_cfg.right_wall_time_period_ratio
        self.wall_colors = geometry_cfg.wall_colors

        self.walls = walls

    def get_start_pos(self):
        """
        randomly generate starting position for tunnel
        """

        # four cases: top, bottom, left, right
        # case = np.random.randint(4)

        # hardcoding to case 2 for now (bottom)
        # x = np.random.randint((self.window_size[0]/2)-300,(self.window_size[0]/2)-200)
        x = self.cfg.start_pos_x #float(self.window_size[0]/2)
        y = 0 #self.cfg.start_pos_y #0 # float(self.window_size[1])
        self.start_pos = [x,y]
        return (x,y)
    
    def save_geometry(self, filename):
        save_walls = []
        for wall in self.walls:
            save_walls.append(wall.save())

        config = {
            'window_size': self.window_size,
            'min_seg_length': self.min_seg_length,
            'max_seg_length': self.max_seg_length,
            'tunnel_width': self.tunnel_width,
            'theta_range': self.theta_range,
            'left_wall_color_time_period': self.left_wall_color_time_period,
            'right_wall_time_period_ratio': self.right_wall_time_period_ratio,
            'wall_colors': self.wall_colors,
            'walls': save_walls,
        }

        j_dict = json.dumps(config)
        f = open("{}.json".format(filename),"w")
        f.write(j_dict)
        f.close()

        # my_df = pd.DataFrame(save_walls)
        # my_df.to_csv('{}.csv'.format(filename), index=False, header=False)

        # with open("{}.csv".format(filename), "wb") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(save_walls)

    def get_end_pos(self, start_pos, segment_len, theta):

        dx = segment_len * np.cos(theta)
        dy = segment_len * np.sin(theta)

        x = start_pos[0] + dx
        y = start_pos[1] + dy

        # edge case
        reached_edge = False
        if x > self.window_size[0]:
            x = self.window_size[0]
            reached_edge = True
        if x < 0:
            x = 0
            reached_edge = True
        if y > self.window_size[1]:
            y = self.window_size[1]
            reached_edge = True
        if y < 0:
            y = 0
            reached_edge = True

        return (x,y), reached_edge, theta

    def alternate_colors(self, walls, freq=1):
        # apply colors to the walls in a tunnel
        wall_colors = []
        for color in self.wall_colors:
            wall_colors.extend([color] * int(freq))

        for i, wall in enumerate(walls):
            wall.color = wall_colors[i % len(wall_colors)]
        return walls
    
    def _create_turn(self, start_pos, segment_len, theta_start=90, theta_final=180, theta_step=5):
        turn_walls = []
        for t in range(theta_start, theta_final, theta_step):
                end_pos, reached_edge, theta = self.get_end_pos(start_pos, self.left_wall_color_time_period, theta)
                seg = Wall(start_pos, end_pos, 'tunnel')
                turn_walls.append(seg)
                start_pos = end_pos
        pass

    def _sample_thetas(self, theta, theta_turn_cnt):
        """
        Samples theta so it causes more uniform turns
        """
        if theta_turn_cnt % 100: 
            theta = np.random.randint(self.theta_range[0], self.theta_range[1])
            theta = np.radians(theta)
        theta_turn_cnt +=1
        return theta, theta_turn_cnt

    def clip_coords(self, x, y):
        x = np.clip(x, 0, self.window_size[0]-5)
        y = np.clip(y, 0, self.window_size[1]-5)
        return [x,y]

    def generate_walls(self, force_regenerate_walls=False):
        """
        main function for creating tunnel
        """
        if (self.walls is not None) and (not force_regenerate_walls): 
            return 
        # else:
        #     print("Generating New Walls...")
        
        left_walls = []
        start_pos = self.get_start_pos()
        self.left_wall_start = list(start_pos)
        self.left_wall_start[1] = 30
        # left walls
        reached_edge = False
        _straight_tunnel_pass_ = self.window_size[1]/4 # first 3/4 should be straight s
        if self.cfg.set_turns:
            _rj_idx = 0
        else: 
            _rj_idx = 100
        
        theta_turn_cnt = 0
        a = np.random.uniform() > 0.5
        if a: 
            tilt = -5. 
        else: 
            tilt = 5
        # tilt = 0

        while not reached_edge:
            segment_len = np.random.randint(self.min_seg_length,self.max_seg_length)
            theta = 90.
            for color_seg in (0, segment_len, self.left_wall_color_time_period):
                if start_pos[1] < _straight_tunnel_pass_ :
                    # so a straight tunnel for some time. 
                    theta = theta + tilt
                    end_pos, reached_edge, _ = self.get_end_pos(start_pos, self.left_wall_color_time_period, np.radians(theta))
                else:
                    if _rj_idx < 10: 
                        if a > 0.5: 
                            # right turn
                            turn_radius = 50. #- (_rj_idx *2)
                            theta = np.radians(turn_radius)
                            if self.cfg.set_new_thetas: 
                                b = np.random.uniform() > 0.6
                                if b:
                                     self.theta_range = [60, 145] # turn the other way 
                        else:
                            # left turn 
                            turn_radius = 40. #- (_rj_idx *2)
                            theta = np.radians(90 + turn_radius)
                            if self.cfg.set_new_thetas: 
                                b = np.random.uniform() > 0.6
                                if b: 
                                    self.theta_range = [50, 100] # turn the other way 
                        end_pos, reached_edge, _ = self.get_end_pos(start_pos, self.left_wall_color_time_period, theta)
                        _rj_idx +=1 
                    else:
                        # go random
                        theta, theta_turn_cnt = self._sample_thetas(theta, theta_turn_cnt)
                        end_pos, reached_edge, theta = self.get_end_pos(start_pos, self.left_wall_color_time_period, theta)

                # print("{}, {}, {}".format(theta, start_pos, end_pos))
                seg = Wall(start_pos, end_pos, 'tunnel')
                left_walls.append(seg)
                start_pos = end_pos

        self.left_wall_end = start_pos
        # self.left_wall_start = left_walls[0].end_pos # first walls end pose 
        left_walls = self.alternate_colors(left_walls)

        # right walls
        right_walls = []
        tunnel_width= self.tunnel_width
        start_pos = left_walls[0].start_pos

        for i, left_wall in enumerate(left_walls):
            # TODO: Sample self.tunnel_width from a distribution
            if self.cfg.sample_tunnel_width: 
                tunnel_width = np.random.randint(self.tunnel_width, self.tunnel_width + 100)
                end_pos = left_wall.end_pos
            else: 
                start_pos = left_wall.start_pos
                end_pos = left_wall.end_pos
                tunnel_width = self.tunnel_width
            sx,sy = self.clip_coords(start_pos[0] + tunnel_width, start_pos[1])
            ex, ey = self.clip_coords(end_pos[0] + tunnel_width, end_pos[1])
            right_wall = Wall((sx,sy), (ex,ey), 'tunnel')
            right_walls.append(right_wall)

            if self.cfg.sample_tunnel_width: 
                start_pos = end_pos
        
        self.right_wall_start = (self.left_wall_start[0] + self.tunnel_width, self.left_wall_start[1])
        self.right_wall_end = (end_pos[0] + tunnel_width, end_pos[1])
        right_walls = self.alternate_colors(right_walls, freq=self.right_wall_time_period_ratio)

        # set goal pos 
        self.goal_start_pos = [self.left_wall_start[0] + self.tunnel_width/2, self.left_wall_start[1]]
        # self.goal_start_pos = np.mean(np.array([self.left_wall_start, self.right_wall_start]), axis=0) 

        self.goal_end_pos = np.mean(np.array([self.left_wall_end, self.right_wall_end]), axis=0) 
        # print("goal_end_pos", self.goal_start_pos, self.goal_end_pos)

        walls = left_walls + right_walls
        
        ## add boundary walls 
        walls.append(Wall((0, 0), (self.window_size[0], 0), type= 'boundary'))
        walls.append(Wall((0, 0), (0, self.window_size[1]), type= 'boundary'))
        walls.append(Wall((self.window_size[0], 0), (self.window_size[0], self.window_size[1]), type= 'boundary'))
        walls.append(Wall((0, self.window_size[1]), (self.window_size[0], self.window_size[1]), type= 'boundary'))
        
        self.walls = walls
         
        # self.save_geometry('geomerty_right_turn')