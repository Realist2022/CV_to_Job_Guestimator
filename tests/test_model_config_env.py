"""Tests for env-var indirection in a configs/llm.yaml model entry.

`base_url_env` / `api_key_env` exist so that a per-workspace endpoint (the
Modal deployment) or a secret can be named by a config that is shared
between developers without the value itself being committed. Constructing
an InstructorClient opens no connection, so these run offline.
"""

import unittest
from unittest.mock import patch

from src.model.adapters import client_from_config


def _config(**overrides) -> dict:
    config = {"provider": "openai_compatible", "model": "cv-guestimator"}
    config.update(overrides)
    return config


class ModelConfigEnvTest(unittest.TestCase):
    @patch.dict(
        "os.environ",
        {"MODAL_INFERENCE_URL": "https://ws--cv-guestimator-serve.modal.run/v1"},
        clear=False,
    )
    def test_base_url_env_resolves_from_environment(self):
        client = client_from_config(_config(base_url_env="MODAL_INFERENCE_URL", api_key="k"))
        self.assertEqual(
            str(client.client.client.base_url).rstrip("/"),
            "https://ws--cv-guestimator-serve.modal.run/v1",
        )

    @patch.dict("os.environ", {"MODAL_INFERENCE_URL": "https://from-env/v1"}, clear=False)
    def test_literal_base_url_wins_over_env(self):
        client = client_from_config(
            _config(
                base_url="http://localhost:11434/v1",
                base_url_env="MODAL_INFERENCE_URL",
                api_key="k",
            )
        )
        self.assertEqual(
            str(client.client.client.base_url).rstrip("/"), "http://localhost:11434/v1"
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_unset_base_url_env_raises_naming_the_variable(self):
        # Not a silent None falling through to the provider default: that
        # surfaces as a connection refused somewhere else entirely.
        with self.assertRaises(ValueError) as caught:
            client_from_config(_config(base_url_env="MODAL_INFERENCE_URL", api_key="k"))
        self.assertIn("MODAL_INFERENCE_URL", str(caught.exception))

    @patch.dict("os.environ", {}, clear=True)
    def test_unset_api_key_env_still_raises(self):
        with self.assertRaises(ValueError) as caught:
            client_from_config(_config(base_url="http://localhost:11434/v1", api_key_env="NOPE"))
        self.assertIn("NOPE", str(caught.exception))

    def test_unrecognised_key_is_still_rejected(self):
        with self.assertRaises(ValueError) as caught:
            client_from_config(
                _config(base_url="http://localhost:11434/v1", api_key="k", base_ur1_env="typo")
            )
        self.assertIn("base_ur1_env", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
