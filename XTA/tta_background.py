"""Cooperative bounds for main-thread background completion work."""
from __future__ import annotations

import time


class BackgroundDrainBudget:
    """Rotate completion categories and return promptly to worker messages.

    A callback is atomic: the budget is checked only after it has settled its
    ownership changes. Credit checkpoints may refill inference but never run a
    second final-result callback inside that transition.
    """

    def __init__(self, checkpoint, *, stages=8, max_completed=8,
                 seconds=.01, clock=time.monotonic):
        if stages < 1 or max_completed < 1 or seconds <= 0:
            raise ValueError('Background drain limits must be positive')
        self.checkpoint = checkpoint
        self.stages, self.max_completed = int(stages), int(max_completed)
        self.seconds, self.clock = float(seconds), clock
        self.cursor = 0
        self.deferred = False
        self.enabled = False

    def begin(self, *, enabled):
        self.enabled = bool(enabled)
        self.start_cursor = self.cursor if self.enabled else 0
        self.completed_count = 0
        self.exhausted = False
        self.deferred = False
        self.started = self.clock()

    def items(self, stage, mapping):
        if self.enabled and stage < self.start_cursor:
            return
        for item in list(mapping):
            if self.enabled and self.exhausted:
                return
            yield item

    def completed(self, stage):
        if not self.enabled:
            return
        self.checkpoint()
        self.completed_count += 1
        if (self.completed_count >= self.max_completed
                or self.clock() - self.started >= self.seconds):
            self.exhausted = True
            self.deferred = True
            self.cursor = (int(stage) + 1) % self.stages

    def finish(self):
        if not self.exhausted:
            # A pass beginning mid-cycle must revisit earlier categories without
            # waiting for a new future notification that may already have fired.
            self.deferred = bool(self.enabled and self.start_cursor)
            self.cursor = 0
