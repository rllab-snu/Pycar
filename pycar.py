# Import a library of functions called 'pygame'
import pygame
import math
import random
import numpy as np
from car_model import Car2

# Define some colors
BLACK = (0,   0,   0)
WHITE = (255, 255, 255)
GREEN = (0, 255,   0)
RED = (255,   0,   0)
BLUE = (0,   0, 255)

PI = math.pi


def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def updateSteering(screen, car):
    pygame.draw.arc(screen, GREEN, [20, 20, 250, 200], PI / 4, 3 * PI / 4, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 3 * PI / 4, PI, 5)
    pygame.draw.arc(screen, RED, [20, 20, 250, 200], 0, PI / 4, 5)
    pygame.draw.circle(screen, BLACK, [145, 120], 20)
    # rotate tip of needle from 145,10
    # centered at 145,120
    x1 = 145 - 145
    y1 = 10 - 120
    x2 = x1 * math.cos(car.steering_angle) - y1 * math.sin(car.steering_angle)
    y2 = x1 * math.sin(car.steering_angle) + y1 * math.cos(car.steering_angle)
    x = x2 + 145
    y = y2 + 120
    pygame.draw.line(screen, BLACK, [x, y], [145, 120], 5)

class Point():
    # constructed using a normal tupple
    def __init__(self, point_t = (0,0)):
        self.x = float(point_t[0])
        self.y = float(point_t[1])
    # define all useful operators
    def __add__(self, other):
        return Point((self.x + other.x, self.y + other.y))
    def __sub__(self, other):
        return Point((self.x - other.x, self.y - other.y))
    def __mul__(self, scalar):
        return Point((self.x*scalar, self.y*scalar))
    def __truediv__(self, scalar):
        return Point((self.x/scalar, self.y/scalar))
    def __len__(self):
        return int(math.sqrt(self.x**2 + self.y**2))
    # get back values in original tuple format
    def get(self):
        return (self.x, self.y)

def draw_dashed_line(surf, color, start_pos, end_pos, width=1, dash_length=10):
    origin = Point(start_pos)
    target = Point(end_pos)
    displacement = target - origin
    length = len(displacement)
    slope = displacement/length

    for index in range(0, int(length/dash_length), 2):
        start = origin + (slope *    index    * dash_length)
        end   = origin + (slope * (index + 1) * dash_length)
        pygame.draw.line(surf, color, start.get(), end.get(), width)

def visualizeTree(env,node):
    sc = 1.0 / env.scale	
    x = sc * node.state[0]
    y = sc * node.state[1]
    #pygame.draw.rect(env.screen,blue,pygame.Rect(x,y,5,5))

    for chd in node.children:
        chd_x = sc * chd.state[0]
        chd_y = sc * chd.state[1]
        pygame.draw.line(env.screen, BLUE, [chd_x, chd_y], [x, y], 1)
        visualizeTree(env,chd)

    if node.parent==None:
        drawRoad(env.screen)
        drawGoal(env.screen, env.goal[0],env.goal[1])
        updateSteering(env.screen, env.car)
        updateSpeedometer(env.screen, env.car)
        #drawCar(env.screen,env.car)
        if env.obs_num!=0:
            for i in range(4):
                drawCar(env.screen,env.obstacles[i])
        pygame.display.flip()

def drawCar(screen,car):
    car_img = car.originalImage.copy()
    car.image = pygame.transform.rotate(car_img, (-car.angle * 360 / (2 * math.pi)))
    car.rect = car.image.get_rect()
    car.rect.center = (car.pose[0],car.pose[1])
    w, h = car.image.get_size()
    screen.blit(car.image,(car.pose[0] - w / 2, car.pose[1] - h / 2))

def drawGoal(screen,x,y):
    pygame.draw.rect(screen,RED,pygame.Rect(x,y,10,10))

def drawRoad(screen):
    # pygame.draw.lines(screen, BLACK, False, [(100,100),(240,100)], 60)

    pygame.draw.lines(screen, BLACK, False, [(50, 300), (1350, 300)], 1)
    draw_dashed_line(screen, BLACK, (50, 350), (1350, 350), 1)
    draw_dashed_line(screen, BLACK, (50, 400), (1350, 400), 1)
    draw_dashed_line(screen, BLACK, (50, 450), (1350, 450), 1)
    pygame.draw.lines(screen, BLACK, False, [(50, 500), (1350, 500)], 1)

    # pygame.draw.arc(screen,BLACK,[210,90,300,300],-PI/2,0,60)
    # pygame.draw.arc(screen,BLACK,[470,100,300,300],0,PI,60)
    # pygame.draw.arc(screen,BLACK,[710,100,300,300],PI,3*PI/2,60)


def updateSpeedometer(screen, car):
    # Select the font to use, size, bold, italics
    font = pygame.font.SysFont('Calibri', 25, True, False)

    # Render the text. "True" means anti-aliased text.
    # Black is the color. This creates an image of the
    # letters, but does not put it on the screen

    if car.gear == "D":
        gear_text = font.render("Gear: Drive", True, BLACK)
    elif car.gear == "STOP":
        gear_text = font.render("Gear: Stopped", True, BLACK)
    elif car.gear == "R":
        gear_text = font.render("Gear: Reverse", True, BLACK)
    else:
        gear_text = font.render("Gear: unknown", True, BLACK)

    # Put the image of the gear_text on the screen
    screen.blit(gear_text, [300, 40])

    speed_text = font.render("Speed: " + str(int(car.speed)/10), True, BLACK)
    screen.blit(speed_text, [300, 60])


def gameLoop(action, car, screen):
    if action == 1 or action == 'a' or action == 'left':
        print('left')
        car.turn(-1)
    elif action == 2 or action == 'd' or action == 'right':
        print('right')
        car.turn(1)


def learningGameLoop():
    print('more code here')

'''
def draw_rrt_path(screen, path):
    for nd in path:
        if(nd.parent != None):
            pygame.draw.line(screen,RED,nd.point,nd.parent.point,1)
'''

class laneFollowingCar1(Car2):
    def __init__(self):
        super().__init__(RED, 60, 385, screen)
        self.car = super().car
        self.car.constant_speed = True
        self.car.speed = 100

class env():
    def __init__(self, visualize=False, discrete=False):

        self.visualize = visualize
        self.discrete = discrete
        self.size = (1400, 600)
        self.scale = 0.001

        self.obs_num = 4

        if visualize:
            # Initialize the game engine
            pygame.init()
            self.screen = pygame.display.set_mode(self.size)
            pygame.display.set_caption("car sim")
            background = pygame.Surface(self.screen.get_size())
            background.fill((0, 0, 0))
            self.clock = pygame.time.Clock()
        else:
            self.screen = None

        start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        goal_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        self.start = (random.uniform(50,100), start_line)
        self.goal = 1300 #(random.uniform(1100,1300), goal_line)
        self.reward = 0
        self.done = False

        self.init_speed = 30
        self.car = Car2(RED, self.start[0], self.start[1], self.screen, self.init_speed)
        self.car.initial_state = (self.start[0], self.start[1], 0, 0, self.init_speed, 0, self.init_speed)
        self.car.speed = self.init_speed
        self.car.vel[0] = self.init_speed
        self.car.reset()

        if self.obs_num!=0:
        
            self.obstacles = [Car2(GREEN, random.uniform(200*(i+1),200*(i+2)), random.uniform(300,500),self.screen) for i in range(4)]

            for obs in self.obstacles:
                obs.gear = 'D'
                obs.speed = random.uniform(10,20)
                #obs.constant_speed = True

            self.obs_state1 = [(self.scale*obs.pose[0], self.scale*obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [self.scale*obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)
        
        self.bound = 40
        self.observation_space = 6
        self.action_space = 2

        self.time_limit = 100
        self.t = 0

        #self.path_planner = rrt_star(self.screen, self.size, self.bound, self.visualize)

        if self.visualize:
            self.screen.fill(WHITE)

    def reset(self):

        start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
        self.start = (50,start_line)#(random.uniform(50,100), start_line)

        self.car.initial_state = (self.start[0], self.start[1], 0, 0, self.init_speed, 0, self.init_speed)
        self.car.speed = self.init_speed
        self.car.vel[0] = self.init_speed
        self.car.reset()

        start_line_shuffle = random.sample(range(4),4)
        for i in range(4):
            speed = random.uniform(10,30)
            start_line = 325+50*start_line_shuffle[i]+random.uniform(-5,5)
            if self.obs_num!=0:
                self.obstacles[i].initial_state = (random.uniform(200*(i+1)+100,200*(i+1)+200),start_line,0,0,speed,0,speed)
                self.obstacles[i].gear="D"
                #self.obstacles[i].constant_speed = True
                self.obstacles[i].reset()
            
        if self.obs_num!=0:
            self.obs_state1 = [(self.scale*obs.pose[0], self.scale*obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [self.scale*obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)

        self.state = np.asarray([self.scale*self.car.pose[0], self.scale*self.car.pose[1], self.scale*self.car.vel[0], self.scale*self.car.vel[1], self.scale*self.car.speed, self.car.steering_angle])

        if self.obs_num!=0:
            self.state = np.concatenate([self.state, np.asarray(self.obs_state)])

        self.reward = 0
        self.done = False
        self.t = 0

        if self.visualize:
            self.screen.fill(WHITE)
            drawRoad(self.screen)
            #drawGoal(self.screen, self.goal[0],self.goal[1])
            #self.road.plotRoad(self.screen)
            updateSteering(self.screen, self.car)
            updateSpeedometer(self.screen, self.car)
            drawCar(self.screen,self.car)
            if self.obs_num!=0:
                for i in range(4):
                    drawCar(self.screen,self.obstacles[i])
            pygame.display.flip()

        return self.state

    def reinitialize(self, new_s):
        angle = math.atan2(new_s[3],new_s[2])

        sc = 1/self.scale

        self.car.initial_state = (sc*new_s[0], sc*new_s[1], angle, new_s[5], sc*new_s[2], sc*new_s[3], sc*new_s[4])
        self.car.reset()

        if self.obs_num!=0:
            for i in range(4):
                speed = sc*new_s[14+i]
                start_line = sc*new_s[7+2*i]
                x = sc*new_s[6+2*i]
                self.obstacles[i].initial_state = (x,start_line,0,0,speed,0,speed)
                self.obstacles[i].gear="D"
                #self.obstacles[i].constant_speed = True
                self.obstacles[i].reset()


        '''
        for i in range(4):
            speed = 10*random.random()
            start_line = 325+50*random.randrange(4)+random.uniform(-5,5)
            if self.obs_num!=0:
                self.obstacles[i].initial_state = (random.uniform(200*(i+1)+100,200*(i+1)+200), start_line,0,0,speed,0,speed)
                self.obstacles[i].gear="D"
                self.obstacles[i].constant_speed = True
                self.obstacles[i].reset()
            
        if self.obs_num!=0:
            self.obs_state1 = [(obs.pose[0], obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)
        '''
        self.state = new_s
        #self.state = (0.01*self.car.pose[0], 0.01*self.car.pose[1], 0.1*self.car.vel[0], 0.1*self.car.vel[1], 0.01*(self.goal[0]-self.car.pose[0]), 0.01*(self.goal[1]-self.car.pose[1]))
        #self.state += self.obs_state if self.obs_num!=0 else ()

        self.reward = 0
        self.done = False
        self.t = 0

        if self.visualize:
            self.screen.fill(WHITE)
            drawRoad(self.screen)
            #drawGoal(self.screen, self.goal[0],self.goal[1])
            #self.road.plotRoad(self.screen)
            updateSteering(self.screen, self.car)
            updateSpeedometer(self.screen, self.car)
            drawCar(self.screen,self.car)
            if self.obs_num!=0:
                for i in range(4):
                    drawCar(self.screen,self.obstacles[i])
            pygame.display.flip()

        return self.state

    
    def step(self, action):
        
        prev_state = self.state
        if (self.discrete):
            steer = 0.5*((action//5)-2)
            accel = 0.5*((action%5)-2)
        else:
            steer = min(1, max(action[0], -1))
            accel = min(2, max(action[1]+1, 0))
        
        self.car.turn(steer)
        self.car.accelerate(accel)

        if self.visualize:
            self.screen.fill(WHITE)
            drawRoad(self.screen)
            #drawGoal(self.screen, self.goal[0],self.goal[1])
            #self.road.plotRoad(self.screen)

        rate = 10
        self.car.update(1 / rate)

        if self.obs_num!=0:
            for obs in self.obstacles:
                obs.update(1 / rate)

        if self.visualize:
            updateSteering(self.screen, self.car)
            updateSpeedometer(self.screen, self.car)
            #draw_rrt_path(self.screen, self.path_planner.path)

            # --- Go ahead and update the screen with what we've drawn.
            pygame.display.flip()

            # --- Limit to 60 frames per second
            self.clock.tick(rate)

        if self.obs_num!=0:
            obs_acc = random.gauss(0,1)
            obs.accelerate(obs_acc)
            self.obs_state1 = [(self.scale*obs.pose[0], self.scale*obs.pose[1]) for obs in self.obstacles]
            self.obs_state2 = [self.scale*obs.speed for obs in self.obstacles]
            self.obs_state = tuple([x for a in self.obs_state1 for x in a] + self.obs_state2)

        self.state = np.asarray([self.scale*self.car.pose[0], self.scale*self.car.pose[1], self.scale*self.car.vel[0], self.scale*self.car.vel[1], self.scale*self.car.speed, self.car.steering_angle])

        if self.obs_num!=0:
            self.state = np.concatenate([self.state, np.asarray(self.obs_state)])

        next_state = self.state
        self.reward, self.done, info = self.reward_check(prev_state, next_state)

        self.t += 1

        return self.state, self.reward, self.done, (info=='reached')

    def is_collide(self, car, obs):

        obs_corner = np.asarray([[0.5*obs.length, 0.5*obs.width]]*4)
        obs_corner[1:,1] -= (1.0*obs.width)
        obs_corner[2:,0] -= (1.0*obs.length)
        obs_corner[3,1] += 1.0*obs.width

        car_corner = obs_corner.copy()
        c = np.cos(car.angle)
        s = np.sin(car.angle)
        rot_mtx = np.asarray([[c,s],[-s,c]])

        car_corner = np.matmul(car_corner,rot_mtx)

        obs_corner[:,0] += obs.pose[0]
        obs_corner[:,1] += obs.pose[1]
        car_corner[:,0] += car.pose[0]
        car_corner[:,1] += car.pose[1]

        for i in range(-1,3):
            for j in range(-1,3):
                A = obs_corner[i]
                B = obs_corner[i+1]
                C = car_corner[j]
                D = car_corner[j+1]
                if intersect(A,B,C,D):
                    return True

        return False

    def collision_check(self):
        if self.obs_num == 0:
            return False

        for obs in self.obstacles:
            if self.is_collide(self.car,obs):
                return True

        return False

    def reward_check(self, prev_s, next_s):

        reward = self.get_reward(prev_s, next_s)

        terminate = False
        info = ''

        if self.done:
            terminate = True
            info = 'done'
        elif self.collision_check():
            terminate = True
            info = 'collision'
        elif(self.car.pose[0]<0):
            terminate = True
            info = 'out of range'
        elif(self.car.pose[1]<300 or self.car.pose[1]>500):
            terminate = True
            info = 'out of range'
        elif self.car.pose[0] > self.goal:
            terminate = True
            info = 'reached'
        elif(self.t >= self.time_limit):
            terminate = True
            info = 'time out'
        else:
            terminate = False
            info = 'on going'

        return reward, terminate, info

    
    def get_reward(self,prev_s,next_s):
        #goal_dist = self.dist(self.car.pose, self.goal)
        goal_dist_prev = abs(prev_s[0] - self.scale*self.goal)
        goal_dist_next = abs(next_s[0] - self.scale*self.goal)

        goal_dist = goal_dist_prev - goal_dist_next

        incline = abs(math.sin(self.car.angle))

        if abs(self.goal-self.car.pose[0]) < self.bound:
            reached = True
        else:
            reached = False

        out_of_range = False
        if(self.car.pose[0]<0):
            out_of_range = True
        elif(self.car.pose[1]<300 or self.car.pose[1]>500):
            out_of_range = True

        collision = self.collision_check()
        #print(10*goal_dist,1*out_of_range,2*collision,5*reached,2*incline)
        return 5 * goal_dist - 2.0 * out_of_range - 2.0 * collision + 5.0 * reached - 0.5 * incline
    
    def dist(self,p1,p2):     #distance between two points
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
        #return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

    def toward_point(self, point):
        x = point[0] - self.car.pose[0]
        y = point[1] - self.car.pose[1]
        th = math.atan2(y,x)       
        
        steering = th - self.car.angle
        steering = min(1, max(steering,-1))
        return steering


if __name__ == "__main__":
    new_env = env(visualize=True)

    test_num = 100
    total_result = 0
    success = 0

    for k in range(test_num):
        
        s = new_env.reset()
        rwd = 0
        for i in range(100):
            speed = 1
          
            steer_control = random.uniform(-1,1)

            acc = 0

            s, r, d, info = new_env.step([steer_control,acc])
            rwd += r
            #print (s)
            #print (r)
            if d:
                if info:
                    success+=1
                break
        print('test {} / reward : {}, success : {}'.format(k,rwd,info))
        total_result+=rwd

    print('total_reward : {}, total_success_rate : {}'.format(total_result/test_num,success/100))
