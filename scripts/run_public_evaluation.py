"""Public hosted 9-scenario DeepSeek evaluation via direct API (no browser)."""

import datetime
import json
import pathlib
import re
import sys
import uuid

import requests

BASE = "https://sentinellab.onrender.com"
OUT = pathlib.Path("reports/public-agent-evaluation-2026-11-21.json")
TIMEOUT = 150


def new_client():
    s = requests.Session()
    s.headers.update({"User-Agent": "sentinellab-eval/1.0"})
    return s


def health(s):
    r = s.get(f"{BASE}/health", timeout=30)
    r.raise_for_status()
    return r.json()


def upload(s, conv_id, path, filename):
    with open(path, "rb") as f:
        files = {"file": (filename, f, "text/csv")}
        r = s.post(
            f"{BASE}/api/upload",
            files=files,
            headers={"X-Conversation-ID": conv_id},
            timeout=60,
        )
    j = r.json()
    if not r.ok or not j.get("ok"):
        raise RuntimeError(f"upload failed {path}: {r.status_code} {j}")
    return j["file"]["reference"]


def chat(s, conv_id, message, condition=None, context_result_id=None):
    body = {"message": message}
    if condition:
        body["condition"] = condition
    if context_result_id:
        body["context_result_id"] = context_result_id
    r = s.post(
        f"{BASE}/api/chat",
        json=body,
        headers={"X-Conversation-ID": conv_id},
        timeout=TIMEOUT,
    )
    try:
        j = r.json()
    except Exception:
        j = {"ok": False, "answer": r.text[:1000], "activities": []}
    return r.status_code, j


def result_detail(s, conv_id, rid):
    r = s.get(
        f"{BASE}/api/results/{rid}",
        headers={"X-Conversation-ID": conv_id},
        timeout=30,
    )
    if not r.ok:
        return None
    return r.json().get("result")


def main():
    s = new_client()
    h = health(s)
    print("HEALTH", json.dumps({k: h.get(k) for k in ("ready", "service")}))
    print("AGENT", h.get("agent"), "MODEL", h.get("model", {}).get("model_version"))
    assert h.get("ready") is True, f"public not ready: {h}"
    model_ver = h["model"]["model_version"]
    provider = h.get("agent", {})

    with open("samples/single-record.json", encoding="utf-8") as fh:
        single = json.load(fh)
    single_str = json.dumps(single)
    cases = []

    def add(
        name,
        conv,
        status,
        resp,
        expected_tools,
        expected_ok=True,
        extra=None,
        prompt=None,
        condition=None,
        context_result_id=None,
        referenced_result=None,
    ):
        tools = [a.get("tool") for a in (resp.get("activities") or [])]
        det = []
        for rid in resp.get("result_ids") or []:
            d = result_detail(s, conv, rid)
            if d:
                det.append(d)
        checks = {
            "tools": tools == expected_tools,
            "success": (resp.get("ok") is True) == expected_ok,
        }
        if extra:
            checks.update(extra(resp, det))
        ok = all(checks.values())
        print(
            f"{name}: http={status} ok={resp.get('ok')} tools={tools} "
            f"behavior_ok={ok} checks={checks}"
        )
        print("  answer:", (resp.get("answer") or "")[:400].replace("\n", " | "))
        cases.append(
            {
                "name": name,
                "conversation_id": conv,
                "request": {
                    "message": prompt,
                    "condition": condition,
                    "context_result_id": context_result_id,
                },
                "referenced_result": referenced_result,
                "http_status": status,
                "expected_tools": expected_tools,
                "actual_tools": tools,
                "expected_success": expected_ok,
                "ok": resp.get("ok"),
                "behavior_ok": ok,
                "checks": checks,
                "error_code": resp.get("error_code"),
                "final_answer": resp.get("answer"),
                "activities": resp.get("activities"),
                "structured_results": det,
                "turns": resp.get("turns"),
                "tool_calls": resp.get("tool_calls"),
            }
        )
        return det

    # AE-01
    c1 = str(uuid.uuid4())
    prompt1 = "Use predict_single for this complete feature record: " + single_str
    st, r1 = chat(
        s,
        c1,
        prompt1,
    )
    add(
        "AE-01 single prediction",
        c1,
        st,
        r1,
        ["predict_single"],
        True,
        lambda resp, det: {
            "hasPrediction": any(d.get("tool") == "predict_single" for d in det)
        },
        prompt=prompt1,
    )

    # AE-02
    c2 = str(uuid.uuid4())
    bref = upload(s, c2, "samples/batch-sample.csv", "batch-sample.csv")
    prompt2 = "Run batch prediction for file reference " + bref + "."
    st, r2 = chat(s, c2, prompt2)
    add(
        "AE-02 batch prediction",
        c2,
        st,
        r2,
        ["predict_batch"],
        True,
        lambda resp, det: {
            "fiveRows": any(
                d.get("tool") == "predict_batch"
                and d.get("data", {}).get("summary", {}).get("input_rows") == 5
                for d in det
            )
        },
        prompt=prompt2,
    )

    # AE-03 threshold met
    c3 = str(uuid.uuid4())
    lref = upload(s, c3, "samples/labeled-sample.csv", "labeled-sample.csv")
    prompt3 = (
        "Evaluate labeled file " + lref + ". If accuracy is at least 98%, "
        "predict this record: " + single_str
    )
    st, r3 = chat(
        s,
        c3,
        prompt3,
        condition={"minimum_accuracy": 98, "evaluation_file_reference": lref},
    )
    add(
        "AE-03 threshold met",
        c3,
        st,
        r3,
        ["evaluate_labeled", "predict_single"],
        True,
        lambda resp, det: {
            "accuracyOne": any(
                d.get("tool") == "evaluate_labeled"
                and d.get("data", {}).get("accuracy") == 1
                for d in det
            )
        },
        prompt=prompt3,
        condition={"minimum_accuracy": 98, "evaluation_file_reference": lref},
    )
    eval_rid = (r3.get("result_ids") or [None])[0]

    # AE-04 threshold not met
    c4 = str(uuid.uuid4())
    fref = upload(s, c4, "samples/evaluation/labels-flipped.csv", "labels-flipped.csv")
    prompt4 = (
        "Evaluate this labeled file and predict the supplied record only if "
        "accuracy is at least 98%. File reference: " + fref + ". Record: " + single_str
    )
    st, r4 = chat(
        s,
        c4,
        prompt4,
        condition={"minimum_accuracy": 98, "evaluation_file_reference": fref},
    )
    add(
        "AE-04 threshold not met",
        c4,
        st,
        r4,
        ["evaluate_labeled"],
        True,
        lambda resp, det: {
            "accuracyZero": any(
                d.get("tool") == "evaluate_labeled"
                and d.get("data", {}).get("accuracy") == 0
                for d in det
            ),
            "noPrediction": not any(
                a.get("tool") == "predict_single" for a in (resp.get("activities") or [])
            ),
        },
        prompt=prompt4,
        condition={"minimum_accuracy": 98, "evaluation_file_reference": fref},
    )

    # AE-05 FN follow-up (same conversation as AE-03, with context)
    prompt5 = "How many false negatives were there in that evaluation?"
    st, r5 = chat(
        s,
        c3,
        prompt5,
        context_result_id=eval_rid,
    )
    referenced = result_detail(s, c3, eval_rid) if eval_rid else None
    add(
        "AE-05 false-negative follow-up",
        c3,
        st,
        r5,
        [],
        True,
        lambda resp, det: {
            "mentionsZero": bool(re.search(r"\b0\b", resp.get("answer") or ""))
        },
        prompt=prompt5,
        context_result_id=eval_rid,
        referenced_result=referenced,
    )

    # AE-06 invalid
    c6 = str(uuid.uuid4())
    iref = upload(s, c6, "samples/evaluation/invalid-feature.csv", "invalid-feature.csv")
    prompt6 = "Use predict_batch for file reference " + iref
    st, r6 = chat(s, c6, prompt6)
    add(
        "AE-06 invalid feature input",
        c6,
        st,
        r6,
        ["predict_batch"],
        False,
        lambda resp, det: {
            "failureSurfaced": bool(
                re.search(r"invalid|valid row|failed", resp.get("answer") or "", re.I)
            )
        },
        prompt=prompt6,
    )

    # AE-07 missing labels
    c7 = str(uuid.uuid4())
    mref = upload(s, c7, "samples/evaluation/missing-label.csv", "missing-label.csv")
    prompt7 = "Evaluate labeled file reference " + mref
    st, r7 = chat(s, c7, prompt7)
    add(
        "AE-07 missing labels",
        c7,
        st,
        r7,
        ["evaluate_labeled"],
        False,
        lambda resp, det: {
            "missingLabelSurfaced": bool(
                re.search(r"label", resp.get("answer") or "", re.I)
            )
        },
        prompt=prompt7,
    )

    # AE-08 single class
    c8 = str(uuid.uuid4())
    sref = upload(s, c8, "samples/evaluation/single-class.csv", "single-class.csv")
    prompt8 = "Evaluate this one-class labeled CSV file reference " + sref
    st, r8 = chat(s, c8, prompt8)
    add(
        "AE-08 single-class evaluation",
        c8,
        st,
        r8,
        ["evaluate_labeled"],
        True,
        lambda resp, det: {
            "aucUndefined": any(
                d.get("tool") == "evaluate_labeled"
                and d.get("data", {}).get("auc") is None
                for d in det
            )
        },
        prompt=prompt8,
    )

    # AE-09 unavailable file
    c9 = str(uuid.uuid4())
    absent = str(uuid.uuid4())
    prompt9 = "Evaluate labeled CSV file reference " + absent
    st, r9 = chat(s, c9, prompt9)
    add(
        "AE-09 unavailable file tool failure",
        c9,
        st,
        r9,
        ["evaluate_labeled"],
        False,
        lambda resp, det: {
            "toolFailureSurfaced": bool(
                re.search(r"failed|unavailable|invalid", resp.get("answer") or "", re.I)
            )
        },
        prompt=prompt9,
    )

    required = [c for c in cases if c["name"].startswith("AE-")]
    passed = sum(1 for c in required if c["behavior_ok"])
    report = {
        "created_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "scope": "public Render deployment via direct API; live DeepSeek provider",
        "base_url": BASE,
        "model_version": model_ver,
        "agent": provider,
        "passed": passed,
        "required": len(required),
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{passed}/{len(required)} scenarios passed; report written to {OUT}")
    return 0 if passed == len(required) else 1


if __name__ == "__main__":
    sys.exit(main())
