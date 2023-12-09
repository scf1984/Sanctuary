from time import sleep
from tkinter import Canvas, Tk

from entities import Bunny
from location import Location
from utils import ilen
from world import World


class WorldRunner:
    def __init__(self, world: World, canvas):
        self.world = world
        self.canvas = canvas
        if self.canvas is not None:
            self.canvas.pack()

    def run(self, n=None):
        from time import time
        loop_tick = 0.25
        prev_time = time()
        i = 0
        while True:
            if n and i >= n:
                break
            i += 1
            now = time()
            dt = now - prev_time
            prev_time = now

            self.world.process_events()
            self.world.update(dt)
            self.world.entity_interactions()
            if self.canvas is not None:
                self.world.render(self.canvas)

            sleep_time = max(loop_tick - (time() - prev_time), 0)
            print(str(sleep_time), ilen(self.world.entities))
            sleep(sleep_time)


if __name__ == '__main__':
    world_size = (500, 500)
    master = Tk()
    master.update()
    w = World(world_size=world_size)
    for _ in range(50):
        Bunny(location=Location.random((0, 500), (0, 500)))
    wr = WorldRunner(world=w, canvas=Canvas(master, width=world_size[0], height=world_size[1]))
    wr.run()

