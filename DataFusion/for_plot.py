import time
import pygame , sys
import numpy as np
import  matplotlib.pyplot as plt

x=[0,1,2,3]
y=[2,1,2,3]

xx=plt.plot(x,y)
plt.show()


"""
pygame.init()

# set up the window with this spicific dim
WINDOW = pygame.display.set_mode((400, 300), 0, 32)

#WINDOWw = pygame.display.set_mode((400, 300), 0, 32)

# the name that appears for the window
pygame.display.set_caption('MPU_6050 Demo')

# set up the colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    time.sleep(0.0005)
    WINDOW.fill(WHITE)
    np.random.rand()*100
    pygame.draw.line(WINDOW,RED,(0 ,0),(100,100),3)
    pygame.draw.line(WINDOW,BLUE,(0 ,100),(100,100),3)
    pygame.display.update()
"""