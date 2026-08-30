from __future__ import annotations

import unittest

from engine.bus import EventBus
from engine.contracts import Event
from engine.validation import validate_output


class BusTests(unittest.IsolatedAsyncioTestCase):
    async def test_publish_fans_out_and_unsubscribes(self) -> None:
        bus = EventBus()
        async with bus.subscribe() as first, bus.subscribe() as second:
            event = Event(kind="test", run_id="run", message="hello")
            await bus.publish(event)
            self.assertIs(await first.get(), event)
            self.assertIs(await second.get(), event)
        self.assertEqual(len(bus.history), 1)

    async def test_publish_does_not_block_when_subscriber_queue_is_full(self) -> None:
        bus = EventBus()
        async with bus.subscribe(maxsize=1) as queue:
            first = Event(kind="one", run_id="run", message="first")
            second = Event(kind="two", run_id="run", message="second")
            await bus.publish(first)
            await bus.publish(second)
            self.assertIs(await queue.get(), first)
            self.assertTrue(queue.empty())
        self.assertEqual(len(bus.history), 2)


class ValidationTests(unittest.TestCase):
    def test_json_shape_and_phrase_criteria(self) -> None:
        results = validate_output('{"tests": true}', ("mentions tests",), "json")
        self.assertTrue(all(result.passed for result in results))

    def test_invalid_json_is_visible(self) -> None:
        results = validate_output("not-json", (), "json")
        self.assertEqual(results[1].criterion, "valid JSON")
        self.assertFalse(results[1].passed)


if __name__ == "__main__":
    unittest.main()
