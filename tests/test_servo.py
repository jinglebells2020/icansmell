"""Tests for ServoLink — serial servo commands, no hardware."""
from __future__ import annotations

from sniffsniff.servo import ServoLink


class _FakeSerial:
    def __init__(self):
        self.written = b""
        self.flushed = 0
        self.closed = False

    def write(self, data):
        self.written += data
        return len(data)

    def flush(self):
        self.flushed += 1

    def close(self):
        self.closed = True


def _link():
    fake = _FakeSerial()
    # startup_delay_s=0 so tests don't sleep; opener injects the fake.
    link = ServoLink("x", startup_delay_s=0.0, opener=lambda: fake)
    return link, fake


def test_move_sends_angle_command():
    link, fake = _link()
    assert link.move(90) == 90
    assert fake.written == b"S90\n"


def test_move_clamps_to_0_180():
    link, fake = _link()
    assert link.move(250) == 180
    assert link.move(-30) == 0
    assert fake.written == b"S180\nS0\n"


def test_opens_once_and_reuses():
    opens = {"n": 0}
    fake = _FakeSerial()

    def opener():
        opens["n"] += 1
        return fake

    link = ServoLink("x", startup_delay_s=0.0, opener=opener)
    link.move(10)
    link.move(20)
    assert opens["n"] == 1  # port opened once, reused


def test_close_is_idempotent_and_closes_serial():
    link, fake = _link()
    link.move(45)
    link.close()
    assert fake.closed is True
    link.close()  # no error second time


def test_context_manager():
    fake = _FakeSerial()
    with ServoLink("x", startup_delay_s=0.0, opener=lambda: fake) as link:
        link.move(120)
    assert fake.written == b"S120\n"
    assert fake.closed is True


def test_does_not_open_in_init():
    opens = {"n": 0}
    ServoLink("x", startup_delay_s=0.0, opener=lambda: opens.__setitem__("n", opens["n"] + 1))
    assert opens["n"] == 0  # construction touches no hardware


def test_cli_servo_one_shot(monkeypatch, capsys):
    from sniffsniff import cli, servo

    moved = {}

    class _FakeLink:
        def __init__(self, port, **kw):
            moved["port"] = port

        def move(self, a):
            moved["angle"] = max(0, min(180, int(a)))
            return moved["angle"]

        def close(self):
            moved["closed"] = True

    monkeypatch.setattr(servo, "ServoLink", _FakeLink)
    rc = cli.main(["servo", "--angle", "135", "--port", "/dev/fake"])
    assert rc == 0
    assert moved == {"port": "/dev/fake", "angle": 135, "closed": True}
    assert "135" in capsys.readouterr().out
