from __future__ import annotations

from types import SimpleNamespace

from app.agent import AgentRunner
from app.agent_config import load_agent_config
from app.agent_protocol import ModelTurn, ScriptedMockProvider
from app.response_guard import validate_final_answer


def test_matching_counts_and_percentages_are_allowed() -> None:
    error = validate_final_answer(
        "Accuracy was 98% with 3 false negatives.",
        [{"accuracy": 0.98, "confusion_matrix": {"fn": 3}}],
    )

    assert error is None


def test_trusted_training_metadata_can_ground_dataset_size_answers() -> None:
    metadata = {
        "training_rows_fitted": 38122,
        "test_rows": 9531,
        "training_fraction": 0.8,
        "test_fraction": 0.2,
    }
    assert (
        validate_final_answer(
            "The model was trained on 38,122 rows and tested on 9,531 rows.",
            [],
            trusted_metadata=metadata,
        )
        is None
    )
    error = validate_final_answer(
        "The model was trained on 40,000 rows and tested on 9,531 rows.",
        [],
        trusted_metadata=metadata,
    )
    assert error is not None and "40000" in error


def test_dataset_description_may_mention_malware_without_claiming_prediction() -> None:
    assert (
        validate_final_answer(
            "The malware dataset has 38,122 training rows and 9,531 test rows.",
            [],
            trusted_metadata={
                "training_rows_fitted": 38122,
                "test_rows": 9531,
                "training_fraction": 0.8,
                "test_fraction": 0.2,
            },
        )
        is None
    )
    assert validate_final_answer("This file is malware.", []) is not None


def test_invented_number_is_rejected() -> None:
    error = validate_final_answer(
        "Accuracy was 99%.",
        [{"accuracy": 0.98}],
    )

    assert error is not None and "99" in error


def test_reasonable_rounding_of_tool_number_is_allowed() -> None:
    error = validate_final_answer(
        "The malware score was 0.9978823131.",
        [{"malware_score": 0.9978823131027981}],
    )

    assert error is None


def test_five_decimal_rounding_of_batch_scores_is_allowed() -> None:
    error = validate_final_answer(
        "Scores ranged from 0.00497 to 0.17922.",
        [{"rows": [{"malware_score": 0.004972218}, {"malware_score": 0.179224334}]}],
    )

    assert error is None


def test_uuid_reference_is_not_treated_as_a_numeric_claim() -> None:
    error = validate_final_answer(
        "Download result 931b042a-ef31-4540-8fc2-9b48b991a232.",
        [{"valid_rows": 5}],
    )

    assert error is None


def test_unsupported_confidence_and_feature_reason_are_rejected() -> None:
    assert validate_final_answer("Confidence is 90%.", [{"malware_score": 0.9}]) is not None
    assert validate_final_answer("Malware because of the feature Entropy.", [{}]) is not None


def test_confidence_claim_is_rejected_but_a_stated_limitation_is_allowed() -> None:
    grounded = [{"accuracy": 1.0, "malware_score": 0.9978823131027981}]

    for claim in (
        "The model is highly confident.",
        "High confidence in the prediction.",
        "I am confident the file is malware.",
        "This result is very confident.",
        "The score provides strong confidence.",
        "The prediction looks confident.",
        "We are confident in this classification.",
    ):
        assert validate_final_answer(claim, grounded) is not None, claim

    for limitation in (
        "The malware_score is an uncalibrated score, not a confidence measure.",
        "This is not a confidence interval; the score is uncalibrated.",
        "The score carries no confidence guarantee.",
        "A confidence level would be misleading here.",
        "Do not read this as confidence.",
        "I am not confident about the outcome.",
    ):
        assert validate_final_answer(limitation, grounded) is None, limitation


def test_numbered_section_headings_are_not_metric_claims() -> None:
    """A step heading such as "Step 2" is presentation, not a metric assertion."""
    answer = (
        "**Step 1 - Evaluation**\n\n"
        "- sample_count: 20\n"
        "- accuracy: 1.0\n\n"
        "**Step 2 - Accuracy gate**\n\n"
        'Your condition was "accuracy at least 98%". The accuracy is 1.0.\n\n'
        "**Step 3 - Prediction**\n\n"
        "- predicted_label: 1\n"
        "- malware_score: 0.9978823131027981\n"
        "- threshold: 0.5"
    )
    grounded = [
        {
            "accuracy": 1.0,
            "sample_count": 20,
            "threshold": 0.5,
            "minimum_accuracy": 0.98,
        },
        {"predicted_label": 1, "malware_score": 0.9978823131027981, "threshold": 0.5},
    ]

    assert validate_final_answer(answer, grounded) is None


def test_numbered_list_items_are_not_metric_claims() -> None:
    answer = "1. Evaluate the file\n2. Predict the record\n\nThe accuracy is 1.0."

    assert validate_final_answer(answer, [{"accuracy": 1.0}]) is None


def test_file_reference_in_a_metric_sentence_is_not_a_claim() -> None:
    """A file reference that starts with digits must not be read as a metric."""
    answer = (
        "- File reference: 5eac940a-5f5d-4f91-abb2-189fd7659e5e\n"
        "- Accuracy: 1.0\n"
        "- Sample count: 20"
    )
    grounded = [{"accuracy": 1.0, "sample_count": 20}]

    assert validate_final_answer(answer, grounded) is None


def test_restated_user_condition_is_grounded_by_the_condition_threshold() -> None:
    """The Agent may repeat the accuracy condition it was given."""
    answer = "**Gate check:** accuracy 1.0 is at least 98%, so the prediction was run."
    grounded = [{"minimum_accuracy": 0.98}, {"accuracy": 1.0}]

    assert validate_final_answer(answer, grounded) is None


def test_invented_metric_is_still_rejected_next_to_a_numbered_heading() -> None:
    answer = "**Step 2 - Results**\n\nThe accuracy was 99%."

    error = validate_final_answer(answer, [{"accuracy": 1.0}])

    assert error is not None and "99" in error


def test_invented_counts_are_still_rejected() -> None:
    grounded = [{"accuracy": 1.0, "confusion_matrix": {"fn": 0, "fp": 0, "tn": 10, "tp": 10}}]

    for answer in (
        "There were 37 false negatives.",
        "The sample count was 250.",
        "The AUC of the uploaded file was 0.87.",
        "The malware score was 0.42.",
        "The threshold was 0.3.",
        "Precision was 0.88 and recall was 0.91.",
    ):
        assert validate_final_answer(answer, grounded) is not None, answer


def test_metric_number_is_rejected_when_it_is_reformatted() -> None:
    """Rounding a metric to a different value is still an invented metric."""
    grounded = [{"accuracy": 0.9801}]

    assert validate_final_answer("The accuracy was 98.01%.", grounded) is None
    assert validate_final_answer("The accuracy was 98%.", grounded) is not None
    assert validate_final_answer("The accuracy was 99%.", grounded) is not None


def test_classification_without_tool_evidence_is_rejected_by_runner() -> None:
    provider = ScriptedMockProvider([ModelTurn(text="This is malware.")])
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        object(),  # type: ignore[arg-type]
    )

    result = agent.run("session", "Classify this")

    assert not result.ok
    assert result.error_code == "ungrounded_response"


def test_clarification_without_classification_is_allowed() -> None:
    provider = ScriptedMockProvider([ModelTurn(text="Which uploaded file should I evaluate?")])
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        object(),  # type: ignore[arg-type]
    )

    result = agent.run("session", "Please evaluate it")

    assert result.ok


def test_runner_answers_dataset_sizes_from_trusted_model_metadata() -> None:
    metadata = {
        "training_rows_fitted": 38122,
        "test_rows": 9531,
        "training_fraction": 0.8,
        "test_fraction": 0.2,
    }
    provider = ScriptedMockProvider(
        [ModelTurn(text="The model was trained on 38,122 rows and tested on 9,531 rows.")]
    )
    tools = SimpleNamespace(model=SimpleNamespace(dataset_metadata=metadata))
    agent = AgentRunner(
        load_agent_config({"LLM_PROVIDER": "mock"}),
        provider,
        tools,  # type: ignore[arg-type]
    )

    result = agent.run("session", "What are the train and test sizes?")

    assert result.ok
    sent_messages = provider.requests[0][0]
    assert any(
        "Trusted, versioned training" in (message.content or "") for message in sent_messages
    )
