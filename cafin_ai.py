"""
cafin_ai.py  --  Amazon Bedrock narration of the trace-clustering results.

Given compact per-cluster statistics (computed by the GUI from the PCA + K-means
clustering of per-cell ΔF/F0 traces), ask an open-source model on Amazon Bedrock to write
a rigorous "findings" narrative — characterising each cluster's dynamics and spatial
location, contrasting them, and proposing testable biological interpretations.

Models: OPEN-SOURCE / open-weights models only, via the model-agnostic **Converse API**.
Default = Meta **Llama 3.3 70B Instruct** — the most capable open-weights model broadly
available on Bedrock and strong at instruction-following/writing for this narrative task.
Fully OSI-open alternatives are offered too: **Mixtral 8x7B** (Apache-2.0) and
**DeepSeek-R1** (MIT). Override `model_id` to use any open model enabled in your account.

Requires: `pip install boto3` and AWS credentials with `bedrock:InvokeModel`
permission for the chosen model (env vars, ~/.aws/credentials, or an IAM role),
plus model access enabled in the Bedrock console for your region.
"""
from __future__ import annotations

import json

DEFAULT_MODEL = "us.meta.llama4-maverick-17b-instruct-v1:0"


def default_region():
    """Use the region configured for the AWS session (e.g. via `aws configure`/`aws login`),
    falling back to us-east-1."""
    try:
        import boto3
        return boto3.Session().region_name or "us-east-1"
    except Exception:
        return "us-east-1"


DEFAULT_REGION = default_region()


def _looks_like_login_expired(err) -> bool:
    s = str(err).lower()
    return ("login" in s and ("expired" in s or "refresh" in s or "reauthenticate" in s)) \
        or "loginrefreshrequired" in s or "token has expired" in s or "expired" in s and "session" in s


def check_credentials(region: str | None = None):
    """Return (ok, message). Validates that AWS credentials resolve and are unexpired
    (via STS get-caller-identity) — cheap, no Bedrock/model cost."""
    region = region or DEFAULT_REGION
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError
    except Exception as e:
        return False, f"boto3 not installed ({e}). Run: pip install boto3"
    try:
        sess = boto3.Session()
        if sess.get_credentials() is None:
            return False, "No AWS credentials found. Run `aws login` (or `aws configure`)."
        ident = boto3.client("sts", region_name=region).get_caller_identity()
        return True, f"AWS OK — account {ident['Account']}, region {region}."
    except NoCredentialsError:
        return False, "No AWS credentials found. Run `aws login` (or `aws configure`)."
    except Exception as e:
        if _looks_like_login_expired(e):
            return False, ("Your AWS login session has EXPIRED. Re-authenticate by running "
                           "`aws login` in a terminal, then retry.")
        return False, f"AWS credential check failed: {type(e).__name__}: {str(e)[:160]}"

# Open-source / open-weights Bedrock model ids for a GUI dropdown (most capable first).
# Cross-region inference profiles need the "us." prefix; the bare id is rejected.
MODEL_CHOICES = [
    "us.meta.llama4-maverick-17b-instruct-v1:0",  # Llama 4 Maverick (17B active, 128 experts)
    "us.meta.llama4-scout-17b-instruct-v1:0",     # Llama 4 Scout (17B active, long context)
    "us.meta.llama3-3-70b-instruct-v1:0",         # Llama 3.3 70B
    "us.meta.llama3-1-70b-instruct-v1:0",         # Llama 3.1 70B
    "us.deepseek.r1-v1:0",                        # DeepSeek-R1 (MIT, reasoning model)
    "mistral.mixtral-8x7b-instruct-v0:1",         # Mixtral 8x7B (Apache-2.0)
    "mistral.mistral-7b-instruct-v0:2",           # Mistral 7B (Apache-2.0)
]

SYSTEM_PROMPT = (
    "You are a quantitative cell biologist analysing single-cell calcium (Ca2+) imaging of the "
    "zebrafish larval fin epithelium. You are given optional researcher-provided background about the "
    "experiment (drug, concentration, treatment, imaging protocol) and per-cluster summary statistics "
    "from an unsupervised clustering (PCA + K-means) of per-cell ΔF/F0 traces. Write a concise, "
    "rigorous 'Findings' narrative that:\n"
    "1. Characterises each cluster by its dynamics — amplitude (mean peak ΔF/F0), transient "
    "frequency, temporal trend (early vs late activity), and within-cluster synchrony — and by its "
    "spatial location in the tissue (mean centroid).\n"
    "2. Contrasts the clusters (which is the most/least active, most synchronised, most localised).\n"
    "3. Proposes 2-3 testable biological interpretations relevant to epithelial Ca2+ signalling and, "
    "where relevant, the specific drug/treatment in the background (e.g. mechanosensitive channels, "
    "gap-junction coupling, wave propagation, cytoskeletal/actin disruption).\n"
    "4. Explicitly states the main uncertainties and what experiment or analysis would confirm each "
    "interpretation.\n\n"
    "Ground the interpretation in the researcher's background when provided. Base every quantitative "
    "statement ONLY on the numbers provided — never invent values. Refer to clusters by their given "
    "id and color. Use short paragraphs with **bold** cluster labels. Keep it under ~450 words."
)

SYSTEM_PROMPT_FULL = (
    "You are a quantitative cell biologist analysing single-cell calcium (Ca2+) imaging of the "
    "zebrafish larval fin epithelium. You are given: (a) optional researcher-provided background "
    "(drug, concentration, treatment, imaging protocol); (b) per-cluster summary statistics; and "
    "(c) the FULL per-cluster and population ΔF/F0 time-series across ALL imaging frames, plus the "
    "per-frame active-cell fraction. Write a rigorous TEMPORAL 'Findings' narrative describing how "
    "activity evolves over the WHOLE recording:\n"
    "1. Overall time-course — baseline, onset/rise, peak timing, decay/plateau, and any late "
    "rebound — for the population and for each cluster.\n"
    "2. How the clusters DIFFER temporally (which leads, which lags, which sustains vs is transient) "
    "and what that implies about coordinated waves or sequential recruitment.\n"
    "3. Relate the time-course to the treatment/protocol in the background (e.g. does activity rise "
    "after drug addition; is the late phase consistent with adaptation/recovery or progressive "
    "cytoskeletal disruption).\n"
    "4. State uncertainties and what analysis/experiment would confirm each claim.\n\n"
    "Ground the interpretation in the researcher's background when provided. Base every statement ONLY "
    "on the numbers/time-series provided — never invent values. Frames are given in order (index 0 = "
    "first). Use short paragraphs with **bold** cluster labels. Keep it under ~500 words."
)


def interpret_clusters(payload: dict, model_id: str = DEFAULT_MODEL,
                       region: str | None = None, max_tokens: int = 1400,
                       temperature: float = 0.4, background: str = "", full: bool = False):
    """Return (ok: bool, text: str). `payload` is the cluster-summary (or full time-series) dict.
    `background` = researcher's free-text context (drug, protocol...). `full=True` uses the
    temporal (all-frames) system prompt."""
    region = region or DEFAULT_REGION
    system_prompt = SYSTEM_PROMPT_FULL if full else SYSTEM_PROMPT
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except Exception as e:
        return False, f"boto3 is not installed ({e}). Run: pip install boto3"

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        bg = ""
        if background and background.strip():
            bg = ("Researcher background about this experiment (use it to ground the interpretation):\n"
                  + background.strip() + "\n\n")
        kind = ("full per-cluster/population time-series across all frames" if full
                else "cluster summary statistics")
        user = (bg + f"Here are the calcium-imaging {kind} as JSON. Write the Findings narrative.\n\n"
                "```json\n" + json.dumps(payload, indent=2) + "\n```")
        cfg = {"maxTokens": max_tokens, "temperature": temperature}

        def _call(with_system):
            if with_system:
                msgs = [{"role": "user", "content": [{"text": user}]}]
                return client.converse(modelId=model_id, system=[{"text": system_prompt}],
                                       messages=msgs, inferenceConfig=cfg)
            # fold the system prompt into the user turn for models that reject `system`
            msgs = [{"role": "user", "content": [{"text": system_prompt + "\n\n" + user}]}]
            return client.converse(modelId=model_id, messages=msgs, inferenceConfig=cfg)

        try:
            resp = _call(with_system=True)
        except Exception as e:
            if "system" in str(e).lower():
                resp = _call(with_system=False)     # model doesn't support a system field
            else:
                raise
        parts = resp["output"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return True, text or "(model returned no text)"
    except NoCredentialsError:
        return False, ("No AWS credentials found. Configure them (aws configure, env vars "
                       "AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or an IAM role).")
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        msg = e.response.get("Error", {}).get("Message", str(e))
        hint = ""
        if code in ("AccessDeniedException", "ValidationException"):
            hint = ("  → Enable access to this model in the Bedrock console for your region, "
                    "and check the model id / IAM permissions.")
        return False, f"Bedrock error [{code}]: {msg}{hint}"
    except BotoCoreError as e:
        if _looks_like_login_expired(e):
            return False, ("Your AWS login session has EXPIRED. Run `aws login` in a terminal to "
                           "re-authenticate, then retry.")
        return False, f"AWS/Bedrock connection error: {e}"
    except Exception as e:
        if _looks_like_login_expired(e):
            return False, ("Your AWS login session has EXPIRED. Run `aws login` in a terminal to "
                           "re-authenticate, then retry.")
        return False, f"Unexpected error calling Bedrock: {e}"


# ------------------------------------------------------------------ follow-up chat
CHAT_SYSTEM = (
    "You are a quantitative cell biologist helping a researcher interpret single-cell calcium "
    "(Ca2+) imaging of the zebrafish larval fin epithelium. The researcher clustered per-cell "
    "ΔF/F0 traces with PCA + K-means. You are given the cluster statistics (and, when available, "
    "the per-cluster time-series across all frames) plus any background the researcher supplied.\n\n"
    "Answer their questions directly and quantitatively. Ground every numeric claim ONLY in the "
    "data given; never invent values. When asked about initiators versus followers, reason from "
    "the timing evidence: a cluster is an initiator if its activity rises earliest (high "
    "activity_early relative to activity_mid/activity_late, or an early peak in its mean trace), "
    "and a follower if it peaks later; cite the numbers you used and name the clusters by id and "
    "colour. Say plainly when the data cannot settle a question, and suggest what analysis or "
    "experiment would. Keep answers focused, a few short paragraphs at most."
)


def chat(payload, history, question, model_id=DEFAULT_MODEL, region=None,
         background="", max_tokens=1200, temperature=0.3):
    """Ask a follow-up question about the clustering results.

    `history` is a list of {"role": "user"|"assistant", "text": ...} from earlier turns.
    Returns (ok, answer_text). The cluster data and background are pinned in the system
    prompt so every turn stays grounded in the same numbers.
    """
    region = region or DEFAULT_REGION
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
    except Exception as e:
        return False, f"boto3 is not installed ({e}). Run: pip install boto3"

    try:
        client = boto3.client("bedrock-runtime", region_name=region)
        ctx = CHAT_SYSTEM
        if background and background.strip():
            ctx += "\n\nResearcher background:\n" + background.strip()
        ctx += "\n\nCluster data (JSON):\n```json\n" + json.dumps(payload, indent=2) + "\n```"

        msgs = [{"role": h["role"], "content": [{"text": h["text"]}]}
                for h in (history or []) if h.get("text")]
        msgs.append({"role": "user", "content": [{"text": question}]})
        cfg = {"maxTokens": max_tokens, "temperature": temperature}

        try:
            resp = client.converse(modelId=model_id, system=[{"text": ctx}],
                                   messages=msgs, inferenceConfig=cfg)
        except Exception as e:
            if "system" in str(e).lower():          # model rejects a system field
                merged = list(msgs)
                merged[0] = {"role": "user",
                             "content": [{"text": ctx + "\n\n" + merged[0]["content"][0]["text"]}]}
                resp = client.converse(modelId=model_id, messages=merged, inferenceConfig=cfg)
            else:
                raise
        parts = resp["output"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return True, text or "(model returned no text)"
    except NoCredentialsError:
        return False, "No AWS credentials found. Run `aws login` (or `aws configure`)."
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        return False, f"Bedrock error [{code}]: {e.response.get('Error', {}).get('Message', str(e))}"
    except (BotoCoreError, Exception) as e:
        if _looks_like_login_expired(e):
            return False, ("Your AWS login session has EXPIRED. Run `aws login` in a terminal, "
                           "then retry.")
        return False, f"Error calling Bedrock: {e}"
