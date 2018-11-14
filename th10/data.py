class Vec2d(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y


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

class Laser(GameObject):

    def __init__(self, **kwargs):
        super(Item, self).__init__(kwargs.get('p', Vec2d(0, 0)),
                                   kwargs.get('v', Vec2d(0, 0)),
                                   kwargs.get('w', 0),
                                   kwargs.get('h', 0))
        self.arc = kwargs.get('arc', 0)
