"""
LLM agent that plays the DualEvade game via OpenRouter.
Supports any model available on OpenRouter.

Failure handling:
  - Exponential backoff with jitter on API errors and 429s
  - On parse failure: defaults to "wait" (safe action)
  - On total API failure after all retries: returns "wait" instead of crashing
  - Context window management: trims old messages if history grows too long
"""
import os
import json
import time
import re
import random
import logging
from typing import Optional, List, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "google/gemini-3-flash-preview"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

MAX_HISTORY_MESSAGES = 40


class LLMAgent:
    """
    An agent that uses an LLM (via OpenRouter) to choose actions in DualEvadeEnv.
    Designed to never crash the experiment — worst case it waits in place.
    """

    def __init__(
        self,
        agent_id: str,
        model: str = DEFAULT_MODEL,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_retries: int = 5,
        api_key: Optional[str] = None,
    ):
        self.agent_id = agent_id
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in env or .env file")

        self.message_history: List[dict] = []
        self.total_tokens = 0
        self.call_count = 0
        self.error_count = 0
        self.parse_failures = 0

    def reset(self):
        self.message_history = []
        if self.system_prompt:
            self.message_history.append({
                "role": "system",
                "content": self.system_prompt,
            })

    def add_context(self, text: str):
        """Add extra context (e.g. episode summary for trust emergence)."""
        self.message_history.append({
            "role": "system",
            "content": text,
        })

    def _trim_history(self):
        """Keep system messages + last N user/assistant pairs to avoid context overflow."""
        system_msgs = [m for m in self.message_history if m["role"] == "system"]
        other_msgs = [m for m in self.message_history if m["role"] != "system"]
        if len(other_msgs) > MAX_HISTORY_MESSAGES:
            other_msgs = other_msgs[-MAX_HISTORY_MESSAGES:]
        self.message_history = system_msgs + other_msgs

    def choose_action(self, state_text: str) -> Tuple[float, float, bool, str]:
        """
        Given a text description of the current state, return (x, y, wait, thoughts).
        Never raises — returns a safe default on failure.
        """
        self._trim_history()

        self.message_history.append({
            "role": "user",
            "content": state_text,
        })

        response_text = self._call_api()

        if response_text is None:
            self.error_count += 1
            return 0.5, 0.5, True, "api_failure"

        self.message_history.append({
            "role": "assistant",
            "content": response_text,
        })

        return self._parse_action(response_text)

    def _call_api(self) -> Optional[str]:
        """Call OpenRouter with exponential backoff + jitter. Returns None on total failure."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nu-snail/Cellworld_game",
        }
        payload = {
            "model": self.model,
            "messages": self.message_history,
            "temperature": self.temperature,
            "max_tokens": 300,
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=60,
                )

                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else (2 ** attempt + random.uniform(0, 1))
                    logger.warning(f"[{self.agent_id}] Rate limited (429), waiting {wait:.1f}s")
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.warning(f"[{self.agent_id}] Server error {resp.status_code}, retry in {wait:.1f}s")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    logger.warning(f"[{self.agent_id}] API returned error: {data['error']}")
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    continue

                usage = data.get("usage", {})
                self.total_tokens += usage.get("total_tokens", 0)
                self.call_count += 1

                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if not content:
                    logger.warning(f"[{self.agent_id}] Empty response from API")
                    time.sleep(1)
                    continue

                return content

            except requests.Timeout:
                wait = 2 ** attempt + random.uniform(0, 1)
                logger.warning(f"[{self.agent_id}] Timeout, retry in {wait:.1f}s (attempt {attempt+1})")
                time.sleep(wait)

            except requests.RequestException as e:
                wait = 2 ** attempt + random.uniform(0, 1)
                logger.warning(f"[{self.agent_id}] Request error: {e}, retry in {wait:.1f}s")
                time.sleep(wait)

            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning(f"[{self.agent_id}] Response parse error: {e}")
                time.sleep(1)

        logger.error(f"[{self.agent_id}] API failed after {self.max_retries} attempts, returning None")
        return None

    def _parse_action(self, text: str) -> Tuple[float, float, bool, str]:
        """
        Parse LLM JSON response into (x, y, wait, thoughts).
        Tries multiple strategies. Never raises.
        """
        # Strategy 1: find a JSON object in the response
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                x = float(data.get("x", 0.5))
                y = float(data.get("y", 0.5))
                wait = bool(data.get("wait", False))
                thoughts = str(data.get("thoughts", ""))
                return max(0.0, min(1.0, x)), max(0.0, min(1.0, y)), wait, thoughts
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Strategy 2: look for coordinate-like numbers in the text
        nums = re.findall(r'(\d+\.\d+)', text)
        if len(nums) >= 2:
            try:
                x = max(0.0, min(1.0, float(nums[0])))
                y = max(0.0, min(1.0, float(nums[1])))
                self.parse_failures += 1
                return x, y, False, f"fallback_parse: {text[:100]}"
            except ValueError:
                pass

        # Strategy 3: check if the model wants to wait
        lower = text.lower()
        if any(w in lower for w in ["wait", "stay", "pause", "observe", "hold"]):
            self.parse_failures += 1
            return 0.5, 0.5, True, f"inferred_wait: {text[:100]}"

        self.parse_failures += 1
        logger.warning(f"[{self.agent_id}] Parse failed, defaulting to wait. Response: {text[:200]}")
        return 0.5, 0.5, True, "parse_failure"

    def action_to_env(self, x: float, y: float, wait: bool):
        """Convert parsed action to the env's continuous action format."""
        import numpy as np
        return np.array([x, y, 1.0 if wait else 0.0], dtype=np.float32)

    def stats(self) -> dict:
        return {
            "model": self.model,
            "agent_id": self.agent_id,
            "calls": self.call_count,
            "total_tokens": self.total_tokens,
            "errors": self.error_count,
            "parse_failures": self.parse_failures,
        }
