"""Message passing enables object impermanent worlds.
"""

class Message:
    """A message can be passed between layers in order to affect the model's
    behavior.

    Each message has a unique id, which can be used to manipulate this message
    afterwards.
    """

    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg

    def __call__(self, layer):
        pass

class ForgetMessage(Message):
    """A forget message tells all layers to forget until it is revoked.
    """

    def __init__(self, cond=None, msg='forget'):
        super().__init__(msg=msg)
        if cond is None:
            cond = lambda x: True
        self.cond = cond

    def __call__(self, layer):
        if self.cond(layer):
            layer.forget()

class MessageStack(dict):
    """A message stack assembles all messages for an atom.
    """

    def __init__(self):
        super().__init__()
        self.new_id = 0

    def add_message(self, msg):
        if isinstance(msg, str):
            msg = Message(msg)
        if not isinstance(msg, Message):
            raise ValueError('msg must be Message but is of type {}.'.\
                             format(type(msg),))
        id = self.get_id()
        self[id] = msg
        return id

    def get_id(self):
        old_id = self.new_id
        self.new_id += 1
        return old_id

    def __call__(self, layer):
        for msg in self.values():
            msg(layer)