"""Run the data build and baseline training on a Lightning AI Studio, then stop it.

    export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=...
    python scripts/lightning_run.py --teamspace <name> --machine T4

Ships the current commit (git archive, so no GitHub token is needed in the Studio),
runs scripts/build_dataset.py and scripts/train_baselines.py for each source, streams the
log, downloads reports/ to --out, and always stops the Studio at the end. It refuses to
start below --min-credits and stops the Studio after --max-hours whatever happens, so a
run can only spend free credits.
"""

from __future__ import annotations

import argparse
import base64
import subprocess
import sys
import time
from pathlib import Path

from lightning_sdk import Machine, Studio, Teamspace
from lightning_sdk.lightning_cloud.rest_client import LightningClient

REMOTE_DIR = "ecoroute"
LOG = f"{REMOTE_DIR}/lightning_run.log"
DONE = f"{REMOTE_DIR}/lightning_run.exit"


def credits(teamspace: Teamspace) -> float:
    client = LightningClient(retry=False)
    return float(client.billing_service_get_project_balance(project_id=teamspace.id).balance)


def ship_code(studio: Studio, chunk: int = 8000) -> None:
    """Copy the current commit into the Studio through its shell.

    Studio.upload_file goes through a presigned storage URL that some networks block, and
    the archive is small, so it is sent as base64 text in a few commands instead.
    """
    archive = subprocess.run(
        ["git", "archive", "--format=tar.gz", "HEAD"], check=True, capture_output=True
    ).stdout
    text = base64.b64encode(archive).decode()
    studio.run("rm -f ~/ecoroute.b64")
    for i in range(0, len(text), chunk):
        studio.run(f"printf %s '{text[i : i + chunk]}' >> ~/ecoroute.b64")
    studio.run(
        f"rm -rf ~/{REMOTE_DIR}/src ~/{REMOTE_DIR}/scripts && mkdir -p ~/{REMOTE_DIR} && "
        f"base64 -d ~/ecoroute.b64 | tar -xz -C ~/{REMOTE_DIR}"
    )


def pipeline(sources: list[str], skip_build: bool) -> str:
    steps = ["pip install -q -e .", "nvidia-smi -L || true"]
    if not skip_build:
        steps.append("python scripts/build_dataset.py --out data/processed")
    steps += [f"python scripts/train_baselines.py --source {s}" for s in sources]
    body = " && ".join(steps)
    return f"cd ~/{REMOTE_DIR} && rm -f ~/{DONE} && ( {body} ) > ~/{LOG} 2>&1; echo $? > ~/{DONE}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teamspace", required=True)
    parser.add_argument("--user", help="Lightning username (defaults to the API key's owner)")
    parser.add_argument("--studio", default="ecoroute")
    parser.add_argument("--machine", default="T4", help="Lightning Machine name, e.g. T4, L4, CPU")
    parser.add_argument("--sources", nargs="+", default=["routerbench", "sprout"])
    parser.add_argument(
        "--skip-build", action="store_true", help="reuse data/processed in the Studio"
    )
    parser.add_argument("--max-hours", type=float, default=2.0)
    parser.add_argument("--min-credits", type=float, default=3.0)
    parser.add_argument("--out", type=Path, default=Path("reports/lightning"))
    args = parser.parse_args()

    user = args.user or LightningClient(retry=False).auth_service_get_user().username
    teamspace = Teamspace(args.teamspace, user=user)
    before = credits(teamspace)
    print(f"credits before: {before:.2f}")
    if before < args.min_credits:
        sys.exit(
            f"only {before:.2f} credits left, below --min-credits {args.min_credits}; not starting"
        )

    studio = Studio(args.studio, teamspace=teamspace, user=user, create_ok=True)
    start = time.time()
    try:
        print(f"starting Studio {args.studio} on {args.machine}")
        studio.start(getattr(Machine, args.machine), max_runtime=int(args.max_hours * 3600))

        ship_code(studio)

        studio.run_and_detach(pipeline(args.sources, args.skip_build), timeout=5)
        shown = 0
        exit_code = None
        while exit_code is None:
            time.sleep(60)
            log = studio.run(f"cat ~/{LOG} 2>/dev/null || true")
            lines = log.splitlines()
            for line in lines[shown:]:
                print(line, flush=True)
            shown = len(lines)
            done = studio.run(f"cat ~/{DONE} 2>/dev/null || true").strip()
            if done:
                exit_code = int(done)
            elif time.time() - start > args.max_hours * 3600:
                print(f"hit --max-hours {args.max_hours}; stopping")
                exit_code = 124

        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "lightning_run.log").write_text(studio.run(f"cat ~/{LOG}"))
        for source in args.sources:
            report = studio.run(f"cat ~/{REMOTE_DIR}/reports/baselines_{source}.json 2>/dev/null")
            if report.strip():
                (args.out / f"baselines_{source}.json").write_text(report)
            else:
                print(f"no report for {source}")
        print(f"pipeline exit code: {exit_code}")
    finally:
        print("stopping Studio")
        studio.stop()
        print(f"elapsed: {(time.time() - start) / 60:.1f} min")
        print(f"credits after: {credits(teamspace):.2f} (before: {before:.2f})")


if __name__ == "__main__":
    main()
