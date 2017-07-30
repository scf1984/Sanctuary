from random import random
from tkinter import Canvas, Tk

from entities import Bunny
from events import ChangeAnimalStateEvent
from location import Location
from states import Walking
from world import World


class Universe(object):
    worlds = []
    pass


class WorldRunner(object):
    def __init__(self, **kwargs):
        self.world = kwargs.get('world')
        self.canvas = kwargs.get('canvas')
        if self.canvas is not None:
            self.canvas.pack()

    def run(self):
        import time
        loop_tick = 0.25
        prev_time = time.time()
        while True:
            now = time.time()
            dt = now - prev_time
            prev_time = now

            self.world.process_events()
            self.world.update(dt)
            self.world.find_animal_interactions()
            if self.canvas is not None:
                self.world.render(self.canvas)

            sleep_time = max(loop_tick - (time.time() - prev_time), 0)
            print(str(sleep_time))
            time.sleep(sleep_time)


if __name__ == '__main__':
    world_size = (500, 500)
    master = Tk()
    master.update()
    w = World(world_map=[], animals=[Bunny(location=Location(250, 250)) for _ in range(50)])
    wr = WorldRunner(world=w, canvas=Canvas(master, width=world_size[0], height=world_size[1]))
    for b in w.animals:
        w.add_event(ChangeAnimalStateEvent(animal=b,
                                           future_state=Walking(b,
                                                          Location.random(
                                                              (0, 500),
                                                              (0, 500)
                                                          ),
                                                          5.0)
                                           , dt=random()*20)
                    )
    wr.run()
