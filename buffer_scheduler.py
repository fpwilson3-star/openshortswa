"""
Buffer GraphQL scheduling module.

- plan_schedule: ask Gemini to order N clips across a posting window and pick post times
- list_channels: fetch user's connected Buffer channels (for UI dropdown)
- submit_post: fire one createPost mutation per (clip, channel) pair
"""

import json
import httpx
from google import genai
from google.genai import types
from gemini_utils import gemini_generate_with_fallback


BUFFER_API_URL = "https://api.buffer.com"
PRESIGNED_URL_EXPIRY = 604800  # 7 days, SigV4 max


SCHEDULE_PROMPT = """\
You are scheduling short-form video clips cut from a podcast episode across a posting window.
The episode itself drops at {episode_drop_iso}. There are {num_clips} clips to schedule across {num_days} day(s) starting then.

Arc principle: the clip MOST DIRECTLY related to the episode's main topic should post FIRST,
near the episode drop, so listeners encountering the clip understand what the episode is about.
Subsequent clips drift toward off-topic, zany, or lighter moments as the week progresses — this
gives the feed variety while anchoring the launch moment to the main idea.

Pick post times based on typical short-form engagement (late morning, lunchtime, and evening
generally outperform; weekends shift slightly later). Spread the clips evenly across the window.

Clips to order (each shown with its AI-generated captions and hook):
{clips_json}

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments):
{{
  "schedule": [
    {{
      "clip_index": <integer matching the input>,
      "post_at_iso": "<ISO 8601 UTC, e.g. 2026-05-22T13:00:00Z>",
      "reasoning": "<one short sentence explaining the placement>"
    }}
  ]
}}

Constraints:
- Every clip in the input MUST appear exactly once.
- post_at_iso values must be in chronological order and within {num_days} day(s) of the episode drop.
- The first scheduled clip should post within ~2 hours of the episode drop.
"""


def plan_schedule(clips, episode_drop_iso, num_days, gemini_api_key):
    """
    Ask Gemini to order clips and pick post times.

    clips: list of dicts; each must include `index` plus the caption fields from metadata.
    Returns: list of {clip_index, post_at_iso, reasoning}.
    """
    client = genai.Client(api_key=gemini_api_key)
    prompt = SCHEDULE_PROMPT.format(
        episode_drop_iso=episode_drop_iso,
        num_clips=len(clips),
        num_days=num_days,
        clips_json=json.dumps(clips, indent=2, ensure_ascii=False),
    )

    response, _ = gemini_generate_with_fallback(
        client,
        [prompt],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    if response is None:
        raise RuntimeError("Gemini failed to plan schedule across all fallback models")

    text = response.text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    data = json.loads(text)
    return data.get("schedule", [])


# Buffer's schema does NOT expose channels under `account` — `channels` is a
# top-level query that requires an organizationId. Org ids come from `account`.
ORGANIZATIONS_QUERY = """
query Organizations {
  account {
    organizations {
      id
    }
  }
}
"""

def _channels_query(org_id):
    """Build the channels query with the org id inlined as a GraphQL literal.

    We inline rather than use a typed variable (the createPost mutation does the
    same) so we don't have to know the schema's exact scalar type for the input —
    a wrong variable type (String! vs ID!) makes Buffer reject the request with 400.
    """
    return (
        "query Channels {\n"
        f"  channels(input: {{ organizationId: {json.dumps(org_id)} }}) {{\n"
        "    id\n"
        "    name\n"
        "    service\n"
        "  }\n"
        "}\n"
    )


def _graphql(client, headers, query):
    """POST a GraphQL query, raise on transport/GraphQL errors, return the data block."""
    resp = client.post(BUFFER_API_URL, headers=headers, json={"query": query})
    if resp.status_code != 200:
        # Surface the response body — Buffer puts the real reason (bad query,
        # missing scope) there, and raise_for_status alone would hide it.
        raise RuntimeError(f"Buffer HTTP {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    if "errors" in body:
        raise RuntimeError(f"Buffer error: {body['errors']}")
    return body.get("data", {}) or {}


def list_channels(buffer_token):
    """Return [{id, name, service}, ...] for the user's connected Buffer channels.

    Buffer scopes channels to an organization, so we first resolve the account's
    organization id(s), then fetch channels for each and combine (deduped by id).
    """
    headers = {
        "Authorization": f"Bearer {buffer_token}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=30) as client:
        orgs = _graphql(client, headers, ORGANIZATIONS_QUERY) \
            .get("account", {}).get("organizations", []) or []
        if not orgs:
            raise RuntimeError("Buffer account has no organizations")

        channels_by_id = {}
        for org in orgs:
            org_id = org.get("id")
            if not org_id:
                continue
            data = _graphql(client, headers, _channels_query(org_id))
            for ch in data.get("channels", []) or []:
                if ch.get("id"):
                    channels_by_id[ch["id"]] = ch

    return list(channels_by_id.values())


def _build_metadata_literal(service, text, title=None):
    """Return the per-platform `metadata` block required by Buffer, or None if not needed."""
    s = (service or "").lower()
    if s == "instagram":
        return "{ instagram: { type: reel, shouldShareToFeed: true } }"
    if s in ("youtube", "youtube_shorts"):
        # YouTube requires both title and categoryId. "22" = People & Blogs.
        # `text` is the post body (becomes the YouTube description); `title` is the
        # video title. They differ for YouTube, so prefer an explicit title and only
        # fall back to text when one isn't supplied.
        title_literal = json.dumps(title or text or "Short")
        return f'{{ youtube: {{ title: {title_literal}, categoryId: "22" }} }}'
    # tiktok and others: no required metadata so far.
    return None


def _build_create_post_mutation(text, channel_id, due_at_iso, video_url, service, thumbnail_url=None, title=None,
                                scheduling_type="automatic"):
    """
    Build the createPost mutation with everything inlined as GraphQL literals.
    GraphQL server infers types from the input shape, so we don't need to know
    internal type names like PostAssetInput.

    scheduling_type: "automatic" (Buffer publishes directly) or "notification"
    (Buffer pings the mobile app at the due time and the user finishes the post
    natively — the only route that allows adding a TikTok sound, since TikTok's
    Content Posting API can't attach audio).
    """
    video_fields = [f"url: {json.dumps(video_url)}"]
    if thumbnail_url:
        video_fields.append(f"thumbnailUrl: {json.dumps(thumbnail_url)}")
    video_literal = "{ " + ", ".join(video_fields) + " }"

    # schedulingType is a GraphQL enum — inline unquoted, whitelisted values only.
    sched = scheduling_type if scheduling_type in ("automatic", "notification") else "automatic"

    input_fields = [
        f"text: {json.dumps(text)}",
        f"channelId: {json.dumps(channel_id)}",
        f"schedulingType: {sched}",
        "mode: customScheduled",
        f"dueAt: {json.dumps(due_at_iso)}",
        f"assets: [{{ video: {video_literal} }}]",
    ]

    metadata = _build_metadata_literal(service, text, title)
    if metadata:
        input_fields.append(f"metadata: {metadata}")

    input_body = ",\n    ".join(input_fields)

    return (
        "mutation CreatePost {\n"
        "  createPost(input: {\n"
        f"    {input_body}\n"
        "  }) {\n"
        "    ... on PostActionSuccess {\n"
        "      post {\n"
        "        id\n"
        "        dueAt\n"
        "      }\n"
        "    }\n"
        "    ... on MutationError {\n"
        "      message\n"
        "    }\n"
        "  }\n"
        "}\n"
    )


def submit_post(buffer_token, channel_id, text, video_url, due_at_iso, service, thumbnail_url=None, title=None,
                scheduling_type="automatic"):
    """
    Submit one scheduled post to Buffer (one channel per call).

    `text` is the post body (the YouTube description / IG-TikTok caption). `title`
    is used only for services that need a separate video title (YouTube); when
    omitted it falls back to `text`. `scheduling_type` "notification" sends a
    finish-in-app reminder instead of auto-publishing (see _build_create_post_mutation).
    Returns {success: bool, post_id?, due_at?, error?}.
    """
    mutation = _build_create_post_mutation(text, channel_id, due_at_iso, video_url, service, thumbnail_url, title,
                                           scheduling_type)

    headers = {
        "Authorization": f"Bearer {buffer_token}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                BUFFER_API_URL,
                headers=headers,
                json={"query": mutation},
            )
    except httpx.HTTPError as e:
        err = f"network: {e}"
        print(f"❌ Buffer submit_post (channel={channel_id}): {err}")
        return {"success": False, "error": err}

    if resp.status_code != 200:
        err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        print(f"❌ Buffer submit_post (channel={channel_id}): {err}")
        return {"success": False, "error": err}

    body = resp.json()
    if "errors" in body:
        err = str(body["errors"])[:500]
        print(f"❌ Buffer submit_post (channel={channel_id}) GraphQL errors: {err}")
        return {"success": False, "error": err}

    create_result = body.get("data", {}).get("createPost", {}) or {}
    if "message" in create_result:
        err = create_result["message"]
        print(f"❌ Buffer submit_post (channel={channel_id}) MutationError: {err}")
        return {"success": False, "error": err}

    post = create_result.get("post") or {}
    return {"success": True, "post_id": post.get("id"), "due_at": post.get("dueAt")}
