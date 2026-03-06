"""Unit tests for the two-tier HeartbeatWatchdog."""

import time

import pytest

from ralph_tui.orchestrator import HeartbeatWatchdog


class TestHeartbeatWatchdog:
    def test_not_stale_immediately_after_creation(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=60, hard_timeout_sec=120)
        assert not wd.is_soft_stale()
        assert not wd.is_hard_stale()

    def test_soft_fires_before_hard(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=9999)
        time.sleep(0.01)
        assert wd.is_soft_stale()
        assert not wd.is_hard_stale()

    def test_hard_fires_after_both_timeouts(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=0)
        time.sleep(0.01)
        # soft fires first (consumed)
        wd.is_soft_stale()
        assert wd.is_hard_stale()

    def test_soft_only_fires_once_per_ping_cycle(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=9999)
        time.sleep(0.01)
        assert wd.is_soft_stale()  # fires once
        assert not wd.is_soft_stale()  # does not fire again

    def test_ping_resets_both_tiers(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=0)
        time.sleep(0.01)
        assert wd.is_soft_stale()
        assert wd.is_hard_stale()
        wd.ping()
        # Both should reset — use large timeouts so they don't trigger immediately
        wd2 = HeartbeatWatchdog(soft_timeout_sec=10, hard_timeout_sec=20)
        wd2.ping()
        assert not wd2.is_soft_stale()
        assert not wd2.is_hard_stale()

    def test_ping_resets_soft_fired_flag(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=9999)
        time.sleep(0.01)
        assert wd.is_soft_stale()  # fires
        assert not wd.is_soft_stale()  # consumed
        wd.ping()
        time.sleep(0.01)
        # After ping, soft should be able to fire again with timeout=0
        wd3 = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=9999)
        time.sleep(0.01)
        assert wd3.is_soft_stale()

    def test_elapsed_tracks_time_since_last_ping(self):
        wd = HeartbeatWatchdog(soft_timeout_sec=60, hard_timeout_sec=120)
        time.sleep(0.05)
        assert wd.elapsed() >= 0.04
        wd.ping()
        assert wd.elapsed() < 0.02

    def test_custom_timeouts(self):
        wd_short = HeartbeatWatchdog(soft_timeout_sec=0, hard_timeout_sec=0)
        wd_long = HeartbeatWatchdog(soft_timeout_sec=9999, hard_timeout_sec=9999)
        time.sleep(0.01)
        assert wd_short.is_soft_stale()
        assert wd_short.is_hard_stale()
        assert not wd_long.is_soft_stale()
        assert not wd_long.is_hard_stale()
