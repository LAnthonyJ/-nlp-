"""Offline evaluation with explicit parsing failures and no model dependencies.

GSM8K mode compares exact rational values without rounding. math-strict mode
accepts exact numeric equality or stripped text equality; it is NOT an official
MATH symbolic-equivalence evaluator. Unmarked model answers are invalid.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import re


def batch_data(items, batch_size=1):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def extract_final_answer(text):
    """Read the last recognized answer marker, balancing nested boxed braces."""
    if not isinstance(text, str):
        return None
    pattern = re.compile(r"The answer is:\s*|####\s*|\\boxed\{")
    position, answer = 0, None
    while (marker := pattern.search(text, position)) is not None:
        start = marker.end()
        if marker.group().startswith("\\boxed"):
            depth = 1
            for end in range(start, len(text)):
                depth += (text[end] == "{") - (text[end] == "}")
                if depth == 0:
                    answer = text[start:end].strip() or None
                    # Nested boxes and markers belong to this outer answer.
                    position = end + 1
                    break
            else:
                return None
        else:
            rest = text[start:]
            answer = rest.splitlines()[0].strip() if rest.strip() else None
            position = start
    return answer


def normalize_answer(text):
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].strip()
    if text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    return text


def parse_number(text):
    if not isinstance(text, str):
        return None
    text = normalize_answer(text)
    latex = re.fullmatch(r"\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}", text)
    if latex:
        text = f"{latex[1]}/{latex[2]}"
    integer = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
    decimal = rf"[+-]?(?:{integer}(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    if not re.fullmatch(rf"{decimal}(?:\s*/\s*{decimal})?", text):
        return None
    try:
        parts = text.replace(",", "").split("/")
        result = Fraction(parts[0].strip())
        return result / Fraction(parts[1].strip()) if len(parts) == 2 else result
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer_number(completion):
    return parse_number(extract_final_answer(completion))


def score_predictions(predictions, references, mode="gsm8k"):
    if mode not in {"gsm8k", "math-strict"}:
        raise ValueError("mode must be gsm8k or math-strict")
    if len(predictions) != len(references):
        raise ValueError("Prediction/reference counts differ; refusing silent truncation")
    records = []
    for index, (prediction, reference) in enumerate(zip(predictions, references)):
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError(f"Reference {index} must be a nonempty string")
        marked_reference = re.search(r"The answer is:|####|\\boxed\{", reference)
        expected = extract_final_answer(reference) if marked_reference else reference
        if not expected:
            raise ValueError(f"Reference {index} has a malformed final answer")
        answer = extract_final_answer(prediction)
        pred_number, ref_number = parse_number(answer), parse_number(expected)
        if mode == "gsm8k" and ref_number is None:
            raise ValueError(f"Reference {index} is not a supported numeric answer")
        invalid = not answer or (mode == "gsm8k" and pred_number is None)
        correct = False
        if not invalid:
            if pred_number is not None and ref_number is not None:
                correct = pred_number == ref_number
            elif mode == "math-strict":
                correct = normalize_answer(answer) == normalize_answer(expected)
        records.append(dict(index=index, prediction=prediction, reference=reference,
                            extracted_answer=answer, correct=correct,
                            status="invalid_output" if invalid else "correct" if correct else "incorrect"))
    correct = sum(item["correct"] for item in records)
    return dict(mode=mode, total=len(records), correct=correct,
                invalid_outputs=sum(r["status"] == "invalid_output" for r in records),
                accuracy=correct / len(records) if records else None, records=records)


def evaluate_generated(prompts, references, predict_batch, batch_size=1, mode="gsm8k"):
    """Backend-neutral generation; the callable must return one string per prompt."""
    if len(prompts) != len(references):
        raise ValueError("Prompt/reference counts differ")
    predictions = []
    for batch in batch_data(prompts, batch_size):
        outputs = predict_batch(batch)
        if not isinstance(outputs, (list, tuple)) or len(outputs) != len(batch):
            raise ValueError("Backend must return one output per prompt")
        if not all(isinstance(item, str) for item in outputs):
            raise ValueError("Backend outputs must be strings")
        predictions.extend(outputs)
    return score_predictions(predictions, references, mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL: prediction and reference string fields")
    parser.add_argument("--mode", choices=["gsm8k", "math-strict"], default="gsm8k")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args(argv)
    try:
        rows = []
        for number, line in enumerate(args.input.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not all(isinstance(row.get(k), str) for k in ("prediction", "reference")):
                raise ValueError(f"Line {number}: prediction and reference must be strings")
            rows.append(row)
        report = score_predictions([r["prediction"] for r in rows], [r["reference"] for r in rows], args.mode)
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
