import math


class Vec2d(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"(x: {self.x}, y: {self.y})"

    def norm(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    def minus(self, other):
        return Vec2d(self.x - other.x, self.y - other.y)


class GameObject(object):

    def __init__(self, p, v, w, h):
        self.p = p
        self.v = v
        self.w = w
        self.h = h


class Item(GameObject):

    def __init__(self, **kwargs):
        super(Item, self).__init__(kwargs.get('p', Vec2d(0, 0)),
                                   kwargs.get('v', Vec2d(0, 0)),
                                   kwargs.get('w', 0),
                                   kwargs.get('h', 0))
        self.item_type = kwargs.get('item_type', 0)


class Player(GameObject):

    def __init__(self, **kwargs):
        super(Player, self).__init__(p=kwargs.get('p', Vec2d(0, 0)),
                                     v=kwargs.get('v', Vec2d(0, 0)),
                                     w=kwargs.get('w', 0),
                                     h=kwargs.get('h', 0))

        """
        status == 0: rebirth state
        status == 1: normal
        status == 2: dead
        status == 3: unknown
        status == 4: being bombed
        """
        prop_defaults = {
            'powers': 0,
            'life': 0,
            'player_type': 0,
            'slow': 0,
            'item_obtain_range': 0,
            'player_status': 0,
            'invincible_time': 0,
        }
        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

    def on_hit(self, enemies):
        for enemy in enemies:
            if abs(enemy.p.x - self.p.x) < 10 and enemy.p.y > self.p.y:
                return 1

        return 0

    def is_near(self, game_objects):
        for game_obj in game_objects:
            if 5. < self.p.minus(game_obj.p).norm() < 20.:
                return 1

        return 0

class Laser(GameObject):

    def __init__(self, **kwargs):
        super(Item, self).__init__(kwargs.get('p', Vec2d(0, 0)),
                                   kwargs.get('v', Vec2d(0, 0)),
                                   kwargs.get('w', 0),
                                   kwargs.get('h', 0))
        self.arc = kwargs.get('arc', 0)
