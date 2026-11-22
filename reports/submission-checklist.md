# Course submission checklist

This checklist prepares the submitter's remaining account and recording steps. It is not a submission receipt. Complete it only after the final deployment and end-to-end gates are accepted.

## Before recording

- [x] Confirm the final project commit and that the repository contains the accepted final artifacts.
- [x] Confirm the public project URL is available without asking the grader to provide a personal LLM key (`https://sentinellab.onrender.com`, `/health ready:true`, Agent `deepseek`/`deepseek-chat` ready).
- [x] Run the final nine-scenario real-LLM evaluation and preserve every scenario result, including failures (`reports/public-agent-evaluation-2026-11-21.json`, 9/9 passed).
- [ ] Rehearse the nine-minute outline in `reports/demo-script.md`; keep the total recording between 8 and 10 minutes.
- [ ] Check that every shown metric agrees with the committed reports.
- [ ] Confirm any team-member appearance and disclosure requirements from the course instructions.

## Submitter account actions

- [ ] Record the demonstration and verify that the uploaded video can be viewed by the intended audience.
- [ ] Add the course grader account (for example, `quantic-grader`) to the private GitHub repository with the required access.
- [ ] Submit both the video URL and repository URL in the course dashboard.
- [ ] Confirm the dashboard reports a successful submission; retain the confirmation or timestamp.
- [ ] If this is a team submission, ensure only the designated member submits and all required members appear in the recording.

## Record completion evidence

After the submitter completes the steps, replace this note with the video URL, repository URL, grader-access confirmation, dashboard submission date, and confirmation evidence. Do not commit credentials or private account information.

Current status: public deployment live at the current release commit; public nine-scenario evaluation passed 9/9 against that deployment; the gated CI/CD pipeline was exercised in both directions (a failing run blocked the deploy job, and the passing run promoted that same commit to the hosted service). Remaining: recording, grader access, and dashboard submission.
