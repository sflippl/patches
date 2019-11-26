"""Tests patches.pilgrimm.messages.
"""

import unittest

from .. import messages as msg

class TestLayer:
    """This layer simply increments a counter every time it is supposed to forget.
    """

    def __init__(self):
        self.i = 0

    def forget(self):
        self.i += 1

class TestMessages(unittest.TestCase):
    """Test if the messages work properly.
    """

    def setUp(self):
        self.dummy_msg = msg.Message('dummy')
        self.forget_msg = msg.ForgetMessage()
        self.cond_forget_msg = msg.ForgetMessage(msg='forget until threshold',
                                                 cond=lambda x: x.i <= 5)

    def test_printer(self):
        self.assertEqual(str(self.dummy_msg), 'dummy')
        self.assertEqual(str(self.forget_msg), 'forget')
        self.assertEqual(str(self.cond_forget_msg), 'forget until threshold')

    def test_forget(self):
        test_layer = TestLayer()
        self.dummy_msg(test_layer)
        self.assertEqual(test_layer.i, 0)
        self.cond_forget_msg(test_layer)
        self.assertEqual(test_layer.i, 1)
        for __ in range(6):
            self.cond_forget_msg(test_layer)
        self.assertEqual(test_layer.i, 6)
        self.forget_msg(test_layer)
        self.assertEqual(test_layer.i, 7)

class TestMessageStacks(TestMessages):
    """Test if the message stacks work properly"""

    def setUp(self):
        super().setUp()
        self.msg_stack = msg.MessageStack()
        self.msg_stack.add_message(self.dummy_msg)
        self.msg_stack.add_message(self.forget_msg)
        self.msg_stack.add_message(self.cond_forget_msg)

    def test_call(self):
        test_layer = TestLayer()
        self.msg_stack(test_layer)
        self.assertEqual(test_layer.i, 2)
        for __ in range(2):
            self.msg_stack(test_layer)
        self.assertEqual(test_layer.i, 6)
        self.msg_stack(test_layer)
        self.assertEqual(test_layer.i, 7)
        self.msg_stack.restart()
        self.msg_stack(test_layer)
        self.assertEqual(test_layer.i, 7)

    def test_fails(self):
        with self.assertRaises(ValueError):
            self.msg_stack.add_message(1)
