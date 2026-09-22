from exhibit.config import CONFIG
from exhibit.domain import FAN_SETTINGS, digest, legacy_policy, run_cache_key
from exhibit.fan_adapter import profiling_argument


def test_encoder_settings_live_only_in_the_policy_registry():
    """demo.json keeps paths, pins and decoder hashes; never an encoder knob."""
    assert "alpha" not in CONFIG
    assert not set(FAN_SETTINGS) & set(CONFIG["fan"])
    policy = legacy_policy()
    assert policy["alpha"] == 0.5
    assert policy["skip"] == -2
    assert policy["skip_pa"] == [0, 1, 2, 3, 4, 5, 6, 7]
    assert policy["use_attn_mask"] is False
    assert profiling_argument(policy) == 0


def test_cache_key_covers_topic_and_personalization():
    base = {"hash": digest("one")}
    other = {"hash": digest("two")}
    assert run_cache_key("cat", base) == run_cache_key("cat", base)
    assert run_cache_key("cat", base) != run_cache_key("tokyo", base)
    assert run_cache_key("cat", base) != run_cache_key("cat", other)
