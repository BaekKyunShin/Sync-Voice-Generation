"""
망설임("어/음/아") 토큰을 한국어 합성 텍스트에 자연스럽게 삽입.

§ 7.4 텍스트 정제 정책으로 비언어 마커는 제거되고 망설임만 보존된 상태가
입력. 여기서는 추가로 합성 단계에서 일부 발화에만 망설임을 더 얹어
TTS 발화 페이스를 자연화한다.

규칙:
- 50% 발화에만 변형 적용 (전체 비율 default)
- 길이별 p 차등 (짧을수록 ↓, 길수록 ↑)
- 위치는 문두/문장경계/쉼표 직후만, 어절 중간 금지
- 이미 어/음/아로 시작하는 발화는 이중 망설임 회피로 스킵
"""
from __future__ import annotations

import random
import re

LEAD_TOKENS = ["어, ", "음, ", "아, ", "그, ", "그러니까, "]
JUNCTION_TOKENS = [" 어 ", " 음... ", " 그... "]
SKIP_PREFIXES = ("어", "음", "아")


def maybe_inject_hesitation(
    text: str, rng: random.Random, apply_rate: float = 0.5
) -> tuple[str, bool]:
    """
    text_clean에 망설임 토큰을 자연스럽게 삽입.

    Args:
        text: 정제된 한국어 텍스트 (text_clean 컬럼).
        rng: 결정론용 random.Random.
        apply_rate: 전체 적용 비율 (0~1).

    Returns:
        (text_synth, applied) — 변형된 텍스트와 실제 적용 여부.
    """
    text = text.strip()
    if not text:
        return text, False

    # 전체 적용 여부
    if rng.random() >= apply_rate:
        return text, False

    # 이미 망설임 시작이면 스킵 (이중 부자연 방지)
    if text.startswith(SKIP_PREFIXES):
        return text, False

    # 길이별 p
    n = len(text)
    if n < 15:
        p_lead, p_junction = 0.4, 0.0
    elif n <= 60:
        p_lead, p_junction = 0.55, 0.55
    else:
        p_lead, p_junction = 0.6, 0.6

    out = text
    applied = False

    # 문두 삽입
    if rng.random() < p_lead:
        out = rng.choice(LEAD_TOKENS) + out
        applied = True

    # 경계 삽입 (마침표/물음표/느낌표/쉼표 직후, 끝부분 제외)
    if n >= 15 and rng.random() < p_junction:
        boundaries = [m.end() for m in re.finditer(r"[.?!,]\s*", out)]
        boundaries = [b for b in boundaries if b < len(out) - 2]
        if boundaries:
            pos = rng.choice(boundaries)
            token = rng.choice(JUNCTION_TOKENS)
            out = out[:pos] + token + out[pos:]
            applied = True

    return out, applied
