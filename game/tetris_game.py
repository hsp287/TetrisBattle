import pygame
import argparse
from TetrisBattle.settings import *

import time as t

from TetrisBattle.renderer import Renderer
from TetrisBattle.utils.features import *
import random
from video_recorder import VideoRecorder
import numpy as np

import torch
from torch import nn
from copy import deepcopy

from TetrisBattle.envs.tetris_interface import ComEvent
from TetrisBattle.tetris import Tetris, Player, Judge, collideDown, collide, collideLeft, collideRight, \
    hardDrop, freeze, get_infos

POS_LIST = [
    {
        'combo': (44, 437),
        'tetris': (314, 477),
        'tspin': (304, 477),
        'back2back': (314, 437),
        'board': (112, 138),
        'drawscreen': (112, 138),
        'big_ko': (44, 235),
        'ko': (140, 233),
        'transparent': (110, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (298, 140, 3, 360),
        'attack_alarm': (298, 481, 3, 18),
        'you_win': (120, 230),
        'you_lose': (115, 230),
    },
    {
        'combo': (415, 437),
        'tetris': (685, 477),
        'tspin': (675, 477),
        'back2back': (685, 437),
        'board': (495, 138),
        'drawscreen': (495, 138),
        'big_ko': (426, 235),
        'ko': (527, 233),
        'transparent': (494, 135),
        'gamescreen': (0, 0), 
        'attack_clean': (680, 140, 3, 360),
        'attack_alarm': (680, 481, 3, 18),
        'you_win': (520, 230),
        'you_lose': (515, 230),
    }
]


def load_agent(agent_type, model_path):
    if agent_type == "multiagent":
        model = MultiAgent()
    elif agent_type == "valuenetwork":
        model = ValueNetwork()
    elif agent_type == "valuenetworkaug":
        model = ValueNetworkAug()
    else:
        model = ValueNetwork()
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def agent_agent_action(tetris, value_net, agent_type):
    # Get all possible states
    final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

    # Compute r(s'|s) + V(s') for each possible state
    future_values = []
    if agent_type == "multiagent":
        for grid, reward in zip(final_states, rewards):
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # For future states, set attacked to 0
            attacked = 0
            info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0)
            value = value_net(grid_tensor, info_vector).item()
            future_values.append(reward + value)
    elif agent_type == "valuenetwork":
        for grid, reward in zip(final_states, rewards):
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            value = value_net(grid_tensor).item()
            future_values.append(reward + value) 
    else:
        future_values = rewards 

    # Select the best future state
    best_index = future_values.index(max(future_values))
    best_action_sequence = action_sequences[best_index]
    
    return best_action_sequence

class MultiAgent(nn.Module):
    def __init__(self):
        super(MultiAgent, self).__init__()
        # Convolutional layers for the grid input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the grid input
        self.fc1_grid = nn.Linear(64 * 10 * 20, 128)
        
        # Fully connected layers for the info vector
        self.fc1_vector = nn.Linear(1, 16)
        
        # Final layers combining both inputs
        self.fc2 = nn.Linear(128 + 16, 128)
        self.fc3 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, grid, vector):
        # Process the grid input
        x_grid = torch.relu(self.conv1(grid))
        x_grid = torch.relu(self.conv2(x_grid))
        x_grid = x_grid.view(x_grid.size(0), -1) 
        x_grid = torch.relu(self.fc1_grid(x_grid))
        
        # Process the vector input
        x_vector = torch.relu(self.fc1_vector(vector))
        
        # Combine both inputs
        x = torch.cat((x_grid, x_vector), dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 20, 128)
        self.fc2 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ValueNetworkAug(nn.Module):
    def __init__(self):
        super(ValueNetworkAug, self).__init__()
        # Convolutional layers for the grid input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the grid input
        self.fc1_grid = nn.Linear(64 * 10 * 20, 128)
        
        # Fully connected layers for the info vector
        self.fc1_vector = nn.Linear(7 * 4 + 2, 64)
        
        # Final layers combining both inputs
        self.fc2 = nn.Linear(128 + 64, 128)
        self.fc3 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, grid, vector):
        # Process the grid input
        x_grid = torch.relu(self.conv1(grid))
        x_grid = torch.relu(self.conv2(x_grid))
        x_grid = x_grid.view(x_grid.size(0), -1) 
        x_grid = torch.relu(self.fc1_grid(x_grid))
        
        # Process the vector input
        x_vector = torch.relu(self.fc1_vector(vector))
        
        # Combine both inputs
        x = torch.cat((x_grid, x_vector), dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class TetrisGame:
    #first function
    #will be used for choosing what map you want to play on

    def __init__(self):
        # import os 
        # os.environ["SDL_VIDEODRIVER"] = "dummy"
        # SCREEN = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT), pygame.FULLSCREEN) # SCREEN is 800*600 
        self.screen = pygame.display.set_mode((SCREENWIDTH, SCREENHEIGHT)) # SCREEN is 800*600 
        images = load_imgs()
        self.renderer = Renderer(self.screen, images)

        self.myClock = pygame.time.Clock() # this will be used to set the FPS(frames/s) 

        self.timer2p = pygame.time.Clock() # this will be used for counting down time in our game

        # whether to fix the speed cross device. Do this by 
        # fix the FPS to FPS (100)
        self._fix_speed_cross_device = True
        self._fix_fps = FPS

    def update_time(self, _time, running):
        # update the time clock and return the running state
        
        if self._fix_speed_cross_device:
            time_per_while = 1 / self._fix_fps * 1000 # transform to milisecond
        else:
            time_per_while = self.timer2p.tick()      # milisecond

        if _time >= 0:                
            _time -= time_per_while * SPEED_UP
        else:
            _time = 0
            running = False

        return _time, running

    def play(self):
        page = "menu"

        while page != "exit":
            if page == "menu":
                page = self.menu(page)
            if page == "start":
                page = self.start()
            if page == "viewmap":
                page = self.viewmap()    
            if page == "instructions":
                page = self.instructions()    
            # if page == "pygame.quit":
            #     page = (page)    
            # if page == "credits":
            #     page = self.credit(page)
        # pygame.quit()

    def setmap(self):
        running = True
        # loading images
        
        # defining rectangles for collision checking
        map1 = pygame.Rect(155, 75, 200, 200)
        map2 = pygame.Rect(439, 75, 200, 200)
        map3 = pygame.Rect(155, 309, 200, 200)
        map4 = pygame.Rect(439, 309, 200, 200)
        buttons = [map1, map2, map3, map4]# preparing a buttons and names list to be zipped together
        names = ["none", "classic", "comboking", "lunchbox"]# 
        self.renderer.drawByName("back1", 0, 0) # IMAGES["back1"] is the main background
        while running:
            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False

            for b, n in zip(buttons, names): # zipping buttons and names together       
                if b.collidepoint(mpos):   # for very easy collision checking            
                    if mb[0] == 1:
                        return n # return the name of the map chosen

            # this chunk of code is just making pretty pictures       
            if map1.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 64)
            elif map2.collidepoint(mpos):
                self.renderer.drawByName("outline", 431, 64)
            elif map3.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            elif map4.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            else:
                self.renderer.drawByName("back1", 0, 0) # keeping it fresh
            
            pygame.display.flip() #necessities
    #gridchoice=""    

    #################################################################################
    def viewmap(self):#viewmap function
        running = True
        #making it look nice
    #defining rectangles for collision checking
        map1 = pygame.Rect(155, 75, 200, 200)
        map2 = pygame.Rect(439, 75, 200, 200)
        map3 = pygame.Rect(155, 309, 200, 200)
        map4 = pygame.Rect(439, 309, 200, 200)
        buttons = [map1, map2, map3, map4]#preparing a buttons and names list to be zipped together
        names = ["none", "classic", "comboking", "lunchbox"]# 
        self.renderer.drawByName("back1", 0, 0) # back1 is the main background
        while running:

            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False

            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            #print mpos

            #for b,n in zip(buttons,names): #zipping buttons and names together       
                #if b.collidepoint(mpos):   #for very easy collision checking            
                    #if mb[0]==1:
                        

            #this chunk of code is just making pretty pictures       
            if map1.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 64)
            elif map2.collidepoint(mpos):
                self.renderer.drawByName("outline", 431, 64)
            elif map3.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            elif map4.collidepoint(mpos):
                self.renderer.drawByName("outline", 149, 301)
            else:
                self.renderer.drawByName("back1", 0, 0) # keeping it fresh

            pygame.display.flip() #necessities

        return "menu"

    #instructions page
    def instructions(self):
        running = True
        self.renderer.drawByName("back2", 0, 0)
        #SCREEN.blit(inst,(173,100))
        while running:
            for evnt in pygame.event.get():          
                if evnt.type == pygame.QUIT:
                    running = False
            pygame.display.flip()

        return "menu"

    #menu page
    def menu(self, page):
        running = True
        button1 = pygame.Rect(320, 204, 146, 50)#start rect
        buttons = [pygame.Rect(325, y * 42 + 275, 135, 30) for y in range(3)]#other three rects 
        vals = ["viewmap", "instructions", "exit"]#values of other three rects
        
        
        self.renderer.drawByName("intro", 0, 0)
        pygame.display.set_caption("Tetris Battle", "tetris battle")
        while running:
            for evnt in pygame.event.get():
                if evnt.type == pygame.QUIT:
                    return "exit"
            mpos = pygame.mouse.get_pos()
            mb = pygame.mouse.get_pressed()
            #print mpos

            #thank u for the code
            #zips button and vals and when
            #a collision and a click occurs
            #returns vals
            for r,v in zip(buttons, vals):
                if r.collidepoint(mpos):
                    #print r,v
                    if mb[0] == 1:
                        return v # page to go to
           
            if button1.collidepoint(mpos):
                self.renderer.drawByName("startbutton", 319, 207)
                if mb[0] == 1:
                    return "start"#starting the game
                #making the game look pretty
            elif buttons[0].collidepoint(mpos):
                self.renderer.drawByName("startbutton", 319, 207)
                self.renderer.drawByName("setmapbutton", 325, 274)
            elif buttons[1].collidepoint(mpos):
                self.renderer.drawByName("helpbutton", 325, 317)
            elif buttons[2].collidepoint(mpos):
                self.renderer.drawByName("quitbutton", 325, 360)
                if mb[0] == 1:
                    return "exit" # quitting the game
            else:
                self.renderer.drawByName("intro", 0, 0) # reblit the background
            
            #draw.rect(SCREEN,(0,0,0),button1,2)
            self.myClock.tick(FPS)            
            pygame.display.flip()

    #main game function
    def start(self):
        raise NotImplementedError
    

class TetrisGameDouble(TetrisGame):

    def __init__(self):
        super(TetrisGameDouble, self).__init__()
        self.num_players = 2

    def start(self):#parameters are FP/s rate and timer countdown
    ################################################################################
        
        gridchoice = self.setmap()#calling the setmap function for a choice of grid
        self.timer2p.tick()
        #the code below is what happens when you set a map
        #different maps = differnet grids

        shared_seed = random.randint(0, 2**32 - 1)
               
        pygame.init() #for music
        battlemusic = pygame.mixer.Sound(MUSIC_PATH)#importing sound file

        #SCREEN=pygame.display.set_mode((800,600))
        running = True #necessity
        # SCREEN.blit(IMAGES["gamescreen"], (0, 0))# blitting the main background
        self.renderer.drawByName("gamescreen", 0, 0)

        #these two used for countdown
        #of the timer
        time = MAX_TIME   # milisecond
        delaytime = time

        info_dict_list = [
            {
            "id": 0,
            "hold": pygame.K_c,
            "drop": pygame.K_SPACE,
            "rotate_right": pygame.K_UP,
            "rotate_left": pygame.K_z,
            "right": pygame.K_RIGHT,
            "left": pygame.K_LEFT,
            "down": pygame.K_DOWN
            },
            {
            "id": 1,
            "hold": pygame.K_e,
            "drop": pygame.K_w,
            "rotate_right": pygame.K_u,
            "rotate_left": pygame.K_q,
            "right": pygame.K_k,
            "left": pygame.K_h,
            "down": pygame.K_j
            }
        ]
        action_meaning = {
            0: "NOOP",
            1: "hold",
            2: "drop",
            3: "rotate_right",
            4: "rotate_left",
            5: "right",
            6: "left",
            7: "down" 
        }
        info_dict = {"id": 1}
        # adding the action information
        for k, v in action_meaning.items():
            info_dict[v] = k

        tetris_list = []

        tetris_list.append({
            'info_dict': info_dict_list[0],
            'tetris': Tetris(Player(info_dict_list[0]), gridchoice, seed=shared_seed),
            'pos': POS_LIST[0]
        })
        tetris_list.append({
            'info_dict': info_dict,
            'tetris': Tetris(Player(info_dict), gridchoice, seed=shared_seed),
            'pos': POS_LIST[1],
            'com_event': ComEvent(),
        })

        winner = 0
        force_quit = 0
        #main loop
        # Initialize variables for the agent's action sequence
        agent_action_sequence = []
        agent_action_index = 0
        agent = load_agent("multiagent", "/home/meriit/TetrisBattle/value_iteration_results/multi/best_model_agent2.pth")

        recorder = VideoRecorder(save_dir="training")
        recorder.start_recording()  # Start recording

        while running:
            # battlemusic.play()#plays music

            recorder.capture_frame(self.screen) 

            for tetris_dict in tetris_list:
                tetris_dict["tetris"].natural_down()

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    force_quit = 1

                tetris_list[0]["tetris"].trigger(evt)  # Player 1 actions
                tetris_list[0]["tetris"].move()

            # Agent logic for the second player
            second_player_tetris = tetris_list[1]["tetris"]

            if agent_action_index >= len(agent_action_sequence):
                print(np.transpose(second_player_tetris.grid))
                agent_action_sequence = agent_agent_action(second_player_tetris, agent, agent_type="multiagent")
                agent_action_index = 0
                #print(agent_action_sequence)

            # Apply the next action in the sequence
            if agent_action_index < len(agent_action_sequence):
                action_agent = agent_action_sequence[agent_action_index]
                if action_agent == 5 or action_agent == 6:
                    if second_player_tetris.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
                        agent_action_index += 1
                elif action_agent == 3 or action_agent == 4:
                    if second_player_tetris.LAST_ROTATE_TIME >= ROTATE_FREQ:
                        agent_action_index += 1
                else:
                    agent_action_index += 1
                com_event = tetris_list[1]["com_event"]
                com_event.set([action_agent])
                for evt in com_event.get():
                    second_player_tetris.trigger(evt)
                tetris_list[1]["tetris"].move()

            for i, tetris_dict in enumerate(tetris_list):
                opponent = tetris_list[self.num_players - 1 - i]
                tetris, pos = tetris_dict["tetris"], tetris_dict["pos"]

                if tetris.check_fallen():
                    # compute the scores and attack the opponent
                    scores = tetris.clear()

                    opponent["tetris"].add_attacked(scores)

                    self.renderer.drawCombo(tetris, *pos["combo"])

                    self.renderer.drawTetris(tetris, *pos["tetris"])
                    self.renderer.drawTspin(tetris, *pos["tspin"])
                    self.renderer.drawBack2Back(tetris, *pos["back2back"])

                    if tetris.check_KO():
                        
                        self.renderer.drawBoard(tetris, *pos["board"])
                        
                        opponent["tetris"].update_ko()

                        tetris.clear_garbage()

                        self.renderer.drawByName("ko", *pos["ko"])
                        self.renderer.drawByName("transparent", *pos["transparent"])

                        # screen.blit(kos[tetris_2.get_KO() - 1], (426, 235))
                        pygame.display.flip()
                        agent_action_index -= 1
                        freeze(0.5)
                        if i == 1:
                            agent_action_sequence = [0] * int(FPS)
                            agent_action_index = 0
                        # scores -= 1

                        # end = 1

                    tetris.new_block()
                    #print(tetris.get_grid()[0])

                    '''
                    # Get all possible states
                    final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

                    # Find the maximum reward
                    max_reward = max(rewards)
                    # Print the results
                    print(f"Number of possible placements: {len(final_states)}")
                    print(f"Highest Reward: {max_reward}")
                    print("-" * 50)

                    for i, (grid, actions, held, reward) in enumerate(zip(final_states, action_sequences, was_held, rewards)):
                        if reward == max_reward:
                            print(f"Placement {i + 1} (Highest Reward):")
                            print(f"Grid:\n{grid}")
                            print(f"Actions: {actions}")
                            print(f"Was Held: {held}")
                            print(f"Reward: {reward}")
                            print("-" * 50)
                    '''


                self.renderer.drawGameScreen(tetris)

                tetris.increment_timer()

                #print(tetris.attacked)
                if tetris.attacked == 0:
                    pygame.draw.rect(self.screen, (30, 30, 30), pos["attack_clean"]) 

                if tetris.attacked != 0:
                    
                    for j in range(tetris.attacked):
                        pos_attack_alarm = list(pos["attack_alarm"])
                        # modified the y axis of the rectangle, according to the strength of attack
                        pos_attack_alarm[1] = pos_attack_alarm[1] - 18 * j
                        pygame.draw.rect(self.screen, (255, 0, 0), pos_attack_alarm) 

                if tetris.KO > 0:
                    self.renderer.drawKO(tetris.KO, *pos["big_ko"])
                    
                self.renderer.drawScreen(tetris, *pos["drawscreen"])

                if Judge.check_ko_win(tetris, max_ko=3):
                    running = False
                    winner = tetris.get_id()

                if Judge.check_ko_win(opponent["tetris"], max_ko=3):
                    running = False
                    winner = opponent["tetris"].get_id()

                time, running = self.update_time(time, running)

                if not running:
                    winner = Judge.who_win(tetris, opponent["tetris"])
                    recorder.stop_and_save("player_vs_agent")

            self.renderer.drawTime2p(time)

            self.myClock.tick(FPS)   
            pygame.display.flip()

        if force_quit:
            return "menu"


        for i, tetris_dict in enumerate(tetris_list):
            opponent = tetris_list[self.num_players - 1 - i]
            tetris, pos = tetris_dict["tetris"], tetris_dict["pos"]
            self.renderer.drawByName("transparent", *pos["transparent"])
            if i == winner: # is winner
                self.renderer.drawByName("you_win", *pos["you_win"])
            else:
                self.renderer.drawByName("you_lose", *pos["you_lose"])
        # # end game
        # # self.renderer.drawByObj(a, 0, 0)
        # self.renderer.drawByName("transparent", 110, 135)
        # self.renderer.drawByName("transparent", 494, 135)

        # if winner == 0:
        #     self.renderer.drawByName("you_win", 120, 230)
        #     self.renderer.drawByName("you_lose", 515, 230)
        # else:
        #     self.renderer.drawByName("you_win", 520, 230)
        #     self.renderer.drawByName("you_lose", 115, 230)

        pygame.display.flip()

        freeze(2.0)

        return "menu"

        # pygame.quit()

class TetrisGameSingle(TetrisGame):

    def __init__(self):
        super(TetrisGameSingle, self).__init__()
        self.num_players = 1

        self.last_infos = {'height_sum': 0, 
                           'diff_sum': 0,
                           'max_height': 0,
                           'holes': 0,
                           'n_used_block': 0}

    def start(self):#parameters are FP/s rate and timer countdown
    ################################################################################

        gridchoice = self.setmap()#calling the setmap function for a choice of grid

        self.timer2p.tick()
        #the code below is what happens when you set a map
        #different maps = differnet grids

        pygame.init() #for music

        battlemusic = pygame.mixer.Sound(MUSIC_PATH)#importing sound file
        #SCREEN=pygame.display.set_mode((800,600))
        running = True #necessity
        self.renderer.drawByName("gamescreen", 0, 0)
        # a = SCREEN.copy()#used for image coverage

        #these two used for countdown
        #of the timer
        time = MAX_TIME  # milisecond
        delaytime = time

        info_dict = {
        "id": 0,
        "hold": pygame.K_c,
        "drop": pygame.K_SPACE,
        "rotate_right": pygame.K_UP,
        "rotate_left": pygame.K_z,
        "right": pygame.K_RIGHT,
        "left": pygame.K_LEFT,
        "down": pygame.K_DOWN
        }

        tetris = Tetris(Player(info_dict), gridchoice)

        pos = POS_LIST[info_dict["id"]]

        opponent_pos = POS_LIST[-1]
        # print("213")
        
        #main loop
        force_quit = 0
        kk = 0
        while running:
            # battlemusic.play()#plays music
            # print(tetris.get_grid()[0])
            
            tetris.natural_down()

            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    force_quit = 1
                tetris.trigger(evt)

            tetris.move()

            if tetris.check_fallen():
                # compute the scores and attack the opponent
                scores = tetris.clear()

                self.renderer.drawCombo(tetris, *pos["combo"])

                self.renderer.drawTetris(tetris, *pos["tetris"])
                self.renderer.drawTspin(tetris, *pos["tspin"])
                self.renderer.drawBack2Back(tetris, *pos["back2back"])

                if tetris.check_KO():
                    
                    self.renderer.drawBoard(tetris, *pos["board"])
                    
                    tetris.clear_garbage()

                    pygame.display.flip()
                    running = False

                tetris.new_block()

            infos = {'is_fallen': tetris.is_fallen}

            if tetris.is_fallen:
                height_sum, diff_sum, max_height, holes = get_infos(tetris.get_board())

                # store the different of each information due to the move
                infos['height_sum'] = height_sum - self.last_infos['height_sum']
                infos['diff_sum'] =  diff_sum - self.last_infos['diff_sum']
                infos['max_height'] =  max_height - self.last_infos['max_height']
                infos['holes'] =  holes - self.last_infos['holes'] 
                infos['n_used_block'] =  tetris.n_used_block - self.last_infos['n_used_block']
                infos['is_fallen'] =  tetris.is_fallen 
                infos['scores'] =  scores 
                infos['cleared'] =  tetris.cleared
                
                self.last_infos = {'height_sum': height_sum,
                                'diff_sum': diff_sum,
                                'max_height': max_height,
                                'holes': holes,
                                'n_used_block': tetris.n_used_block}
                
                
                # Get all possible states
                final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

                # Find the maximum reward
                max_reward = max(rewards)

                #print(tetris.get_grid()[1])
                '''
                # Print the results
                print(f"Number of possible placements: {len(final_states)}")
                print(f"Highest Reward: {max_reward}")
                print("-" * 50)

                for i, (grid, actions, held, reward) in enumerate(zip(final_states, action_sequences, was_held, rewards)):
                    if reward == max_reward:
                        print(f"Placement {i + 1} (Highest Reward):")
                        print(f"Grid:\n{grid}")
                        print(f"Actions: {actions}")
                        print(f"Was Held: {held}")
                        print(f"Reward: {reward}")
                        print("-" * 50)
                '''
                

            self.renderer.drawGameScreen(tetris)

            self.renderer.drawScreen(tetris, *pos["drawscreen"])

            self.renderer.drawByName("transparent", *opponent_pos["transparent"])

            tetris.increment_timer()

            # pygame.display.update(r)

            time, running = self.update_time(time, running)

            self.renderer.drawTime2p(time)

            # time goes until it hits zero
            # when it hits zero return endgame SCREEN
            self.myClock.tick(FPS)   
            pygame.display.flip()

        if force_quit:
            return "menu"
        # self.renderer.drawByObj(a, 0, 0)
        self.renderer.drawByName("transparent", *pos["transparent"])

        pygame.display.flip()
        freeze(2.0)
        # pygame.quit()
        return "menu"

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "double"], default="single")

    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    if args.mode == "single":
        game = TetrisGameSingle()
    else:
        game = TetrisGameDouble()
    game.play()