# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from mathruler.grader import grade_answer


def retrieve_format_reward(predict_str: str) -> float:
    try:
        predict_str = predict_str.strip()

        overall_pattern = re.compile(
            r"<think>.*</think>.*Answer:.*", re.DOTALL)
        if not overall_pattern.fullmatch(predict_str):
            return 0.0

        think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        think_match = think_pattern.search(predict_str)

        if not think_match:
            return 0.0

        think_content = think_match.group(1)
        retrieval_pattern = re.compile(
            r"<retrieval>.*?</retrieval>", re.DOTALL)
        retrieval_matches = retrieval_pattern.findall(think_content)

        if not retrieval_matches:
            return 0.0

        return 1.0
    except Exception:
        return 0.0


def retrieve_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    try:
        content_match = re.search(r"Answer:(.*)", predict_str, re.DOTALL)
        given_answer = content_match.group(
            1).strip() if content_match else predict_str.strip()
        if grade_answer(given_answer, ground_truth.strip()):
            return 1.0

    except Exception:
        pass

    return 0.0


def retrieval_spans_in_context(predict_str: str, context: str) -> float:
    spans = re.findall(r"<retrieval>(.*?)</retrieval>", predict_str, re.DOTALL)
    if not spans:
        return 0.0

    spans_found = 0
    for span in spans:
        cleaned_span = re.sub(r'\s+', ' ', span).strip()
        if not cleaned_span:
            continue
        if cleaned_span in context:
            spans_found += 1

    if not spans:
        return 0.0
    return min(1.0, spans_found / len([s for s in spans if s.strip()]))


def compute_score(predict_str: str, ground_truth: str, context: str, format_weight=0.1, retrieval_weight=0.2) -> float:
    format_score = retrieve_format_reward(predict_str)
    accuracy_score = retrieve_accuracy_reward(predict_str, ground_truth)
    retrieval_score = retrieval_spans_in_context(predict_str, context)

    return {
        "overall": (1 - format_weight - retrieval_weight) * accuracy_score + format_weight * format_score + retrieval_weight * retrieval_score,
        "format": format_score,
        "accuracy": accuracy_score,
        "retrieval": retrieval_score,
    }
