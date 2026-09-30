"""GiGPO (Group-in-Group Policy Optimization) for OSWorld online RL.

Ported from ClawGUI / verl-agent's GiGPO implementation
(``clawgui-rl/gigpo/core_gigpo.py``, arXiv 2505.10978). Claws use the
environment's textual feedback as the per-step "anchor observation" key for
grouping; on OSWorld the analogue is the GUI's **accessibility tree**. The
default hybrid anchor first matches a hierarchy/state-aware canonical hash,
then merges only when no semantic conflict signal is present. The previous
similarity-threshold hybrid is preserved in ``gigpo_similarity_legacy.py``.

This module is intentionally self-contained: it depends only on the stdlib,
``numpy``, and ``torch``. It does NOT import OSWorld internals (env_infra's
``mm_agents`` are read-only vendor code and are not on this package's
PYTHONPATH). The a11y normalizer re-implements the essential subset of OSWorld's
``linearize_accessibility_tree`` (``mm_agents/agent.py:71``) plus the two extra
steps the vendor linearizer does not do — sorting nodes and dropping
coordinates — which are what turn the raw XML into a deterministic hash key.

Algorithm (separate trajectory + state advantage variant):
  1. trajectory advantage: normalize each trajectory outcome ``R_traj`` among
     the prompt's rollouts. Batch-wide dynamic-history length scaling is applied
     later by Slime, once all prompt groups are available.
  2. discounted analyzer return: within each trajectory, accumulate only the
     analyzer's per-step rewards as ``Q_t = r_t + gamma * Q_{t+1}``.
  3. anchor-state grouping: within an episode group, cluster step-samples that
     reached the same normalized a11y state; assign each cluster a uid.
  4. state advantage: normalize ``Q_t`` within each anchor sub-group. A
     single-element sub-group receives zero step advantage because no
     state-relative comparison exists.
  5. combine advantages: ``A_t = A_traj_scaled + w * A_step``.

Integration contract: ``compute_gigpo_advantages`` consumes a ``list[Sample]``
of one prompt group's rollouts that have already been expanded to per-step
training samples (build_dynamic_history_samples). Each sample must already
carry ``metadata["gigpo"] = {anchor_obs, anchor_hash, step_index}`` and
``metadata["step_wise"]["step_scores"]`` / ``["outcome_reward"]`` (written by
the rollout + reward pipeline). The returned scalar is meant to be placed into
``sample.reward["score"]`` while ``--disable-rewards-normalization`` keeps
slime from re-normalizing it, so ``get_grpo_returns`` broadcasts the GiGPO
advantage to the step's response tokens unchanged.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
import uuid
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Accessibility-tree normalizer                                               #
# --------------------------------------------------------------------------- #

# Namespace URIs the OSWorld server writes into the a11y XML
# (desktop_env/server/main.py:370-403); the OSWorld-native filter/linearize
# copy below uses the same URIs (named _state_ns_* / _component_ns_* / ...) so
# attribute lookups match the server's output byte-for-byte.

# --------------------------------------------------------------------------- #
# OSWorld-native a11y filtering + state-key linearization                      #
# --------------------------------------------------------------------------- #
# ``_judge_node`` and ``_filter_nodes`` are self-contained copies of OSWorld's
# vendor code (env_infra/OSWorld/mm_agents/accessibility_tree_wrap/
# heuristic_retrieve.py). They are copied rather than imported because
# (1) OSWorld is a read-only vendor submodule (env_infra/CLAUDE.md:34 — DO NOT
# modify internal code), and (2) gui-rl's PYTHONPATH deliberately excludes
# env_infra (it keeps self-contained copies of env_infra helpers, e.g.
# clients/osworld_remote_async.py). Keeping this faithful to the vendor path
# The state-key path starts with that vendor filter, then adds textual editable
# document descendants so offscreen edits can participate when the server
# exposes them.
#
# The vendor linearizer alone is NOT a usable anchor key: it keeps the jittering
# position/size columns and emits nodes in (non-deterministic) document order.
# a11y_anchor_key below adds the three steps OSWorld omits — drop coords,
# canonical-sort, hash — to turn the linearized text into a stable state key.

_state_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/state"
_state_ns_windows = "https://accessibility.windows.example.org/ns/state"
_component_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/component"
_component_ns_windows = "https://accessibility.windows.example.org/ns/component"
_value_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/value"
_value_ns_windows = "https://accessibility.windows.example.org/ns/value"
_attributes_ns_ubuntu = "https://accessibility.ubuntu.example.org/ns/attributes"
_attributes_ns_windows = "https://accessibility.windows.example.org/ns/attributes"
_class_ns_windows = "https://accessibility.windows.example.org/ns/class"


def _judge_node(node: ET, platform: str = "ubuntu", check_image: bool = False) -> bool:
    """Verbatim copy of OSWorld heuristic_retrieve.judge_node (lines 38-91).

    Decides whether an a11y node is kept: tag allow-list AND on-screen visibility
    AND interactive AND (has name/text), with sane screen coordinates. This
    visibility gate is what removes the cross-VM jitter (clock widgets, hidden
    panels, transient notifications) that a hand-rolled tag filter misses.
    """
    if platform == "ubuntu":
        _state_ns = _state_ns_ubuntu
        _component_ns = _component_ns_ubuntu
    elif platform == "windows":
        _state_ns = _state_ns_windows
        _component_ns = _component_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    keeps: bool = node.tag.startswith("document") \
        or node.tag.endswith("item") \
        or node.tag.endswith("button") \
        or node.tag.endswith("heading") \
        or node.tag.endswith("label") \
        or node.tag.endswith("scrollbar") \
        or node.tag.endswith("searchbox") \
        or node.tag.endswith("textbox") \
        or node.tag.endswith("link") \
        or node.tag.endswith("tabelement") \
        or node.tag.endswith("textfield") \
        or node.tag.endswith("textarea") \
        or node.tag.endswith("menu") \
        or node.tag in {"alert", "canvas", "check-box", "combo-box", "entry",
                        "icon", "image", "paragraph", "scroll-bar", "section",
                        "slider", "static", "table-cell", "terminal", "text",
                        "netuiribbontab", "start", "trayclockwclass",
                        "traydummysearchcontrol", "uiimage", "uiproperty",
                        "uiribboncommandbar"}
    keeps = keeps and (
        (platform == "ubuntu"
         and node.get("{{{:}}}showing".format(_state_ns), "false") == "true"
         and node.get("{{{:}}}visible".format(_state_ns), "false") == "true")
        or (platform == "windows"
            and node.get("{{{:}}}visible".format(_state_ns), "false") == "true")
    ) and (
        node.get("{{{:}}}enabled".format(_state_ns), "false") == "true"
        or node.get("{{{:}}}editable".format(_state_ns), "false") == "true"
        or node.get("{{{:}}}expandable".format(_state_ns), "false") == "true"
        or node.get("{{{:}}}checkable".format(_state_ns), "false") == "true"
    ) and (
        node.get("name", "") != "" or (node.text is not None and len(node.text) > 0)
        or (check_image and node.get("image", "false") == "true")
    )

    coordinates: Tuple[int, int] = eval(node.get("{{{:}}}screencoord".format(_component_ns), "(-1, -1)"))  # noqa: S307 - vendor parity
    sizes: Tuple[int, int] = eval(node.get("{{{:}}}size".format(_component_ns), "(-1, -1)"))  # noqa: S307 - vendor parity
    keeps = keeps and coordinates[0] >= 0 and coordinates[1] >= 0 and sizes[0] > 0 and sizes[1] > 0
    return keeps


def _filter_nodes(root: ET, platform: str = "ubuntu", check_image: bool = False):
    """Verbatim copy of OSWorld filter_nodes (heuristic_retrieve.py:94-102)."""
    filtered_nodes = []
    for node in root.iter():
        if _judge_node(node, platform, check_image):
            filtered_nodes.append(node)
    return filtered_nodes


def _is_gnome_shell_notification_node(
    node: ET.Element,
    parent_by_node: dict[ET.Element, ET.Element],
) -> bool:
    """Whether a node belongs to GNOME Shell's automatic notification tray."""
    current: ET.Element | None = node
    under_notification = False
    while current is not None:
        if current.tag == "notification":
            under_notification = True
        if current.tag == "application":
            return under_notification and _clean_text(current.get("name")) == "gnome-shell"
        current = parent_by_node.get(current)
    return False


def _semantic_filter_nodes(root: ET, platform: str = "ubuntu"):
    """Return vendor-filtered nodes plus textual editable document content.

    OSWorld's interaction filter intentionally keeps only currently visible
    nodes.  That is appropriate for action retrieval, but it loses state when
    an edit was made on another document page.  For state identity, retain
    text-bearing editable document descendants even when their viewport
    visibility or coordinates fail the interaction filter.  The server must
    expose the content for this to help; this does not invent missing nodes.
    """
    parent_by_node = {child: parent for parent in root.iter() for child in parent}
    kept = [
        node
        for node in _filter_nodes(root, platform)
        if not _is_gnome_shell_notification_node(node, parent_by_node)
    ]
    seen = {id(node) for node in kept}
    state_ns = _state_ns_ubuntu if platform == "ubuntu" else _state_ns_windows
    for node in root.iter():
        if id(node) in seen:
            continue
        ancestor = parent_by_node.get(node)
        in_document = node.tag.startswith("document")
        while ancestor is not None and not in_document:
            in_document = ancestor.tag.startswith("document")
            ancestor = parent_by_node.get(ancestor)
        if not in_document:
            continue
        text = (node.text or "").strip()
        editable = node.get(f"{{{state_ns}}}editable") == "true"
        # Text is the content signal; a named container such as a footer frame
        # is not itself document content and should not create a false state.
        if editable and text:
            kept.append(node)
    return kept


def _linearize_accessibility_tree(
    accessibility_tree: str, platform: str = "ubuntu", *, keep_coords: bool = False,
) -> List[str]:
    """Linearize nodes for a stable state key.

    This follows OSWorld's columns while also retaining textual editable
    document descendants that are outside the current viewport. Coordinates
    remain optional because they jitter on window move/resize.
    """
    if platform == "ubuntu":
        _attributes_ns = _attributes_ns_ubuntu
        _value_ns = _value_ns_ubuntu
    elif platform == "windows":
        _attributes_ns = _attributes_ns_windows
        _value_ns = _value_ns_windows
        _class_ns = _class_ns_windows
    else:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")

    filtered_nodes = _semantic_filter_nodes(ET.fromstring(accessibility_tree), platform)
    rows: List[str] = []

    for node in filtered_nodes:
        if node.text:
            text = node.text if '"' not in node.text \
                else '"{:}"'.format(node.text.replace('"', '""'))
        elif platform == "windows" \
                and node.get("{{{:}}}class".format(_class_ns), "").endswith("EditWrapper") \
                and node.get("{{{:}}}value".format(_value_ns)):
            node_text = node.get("{{{:}}}value".format(_value_ns), "")
            text = node_text if '"' not in node_text \
                else '"{:}"'.format(node_text.replace('"', '""'))
        else:
            text = '""'

        cols = [
            node.tag,
            node.get("name", ""),
            text,
            node.get("{{{:}}}class".format(_attributes_ns), "") if platform == "ubuntu"
            else node.get("{{{:}}}class".format(_class_ns), ""),
            node.get("{{{:}}}description".format(_attributes_ns), ""),
        ]
        if keep_coords:
            _component_ns = _component_ns_ubuntu if platform == "ubuntu" else _component_ns_windows
            cols.append(node.get('{{{:}}}screencoord'.format(_component_ns), ""))
            cols.append(node.get('{{{:}}}size'.format(_component_ns), ""))
        rows.append("\t".join(cols))
    return rows


def normalize_a11y(
    a11y_xml: str | None,
    *,
    keep_coords: bool = False,
    keep_text: bool = True,
    platform: str = "ubuntu",
) -> str:
    """Turn a raw accessibility-tree XML string into a deterministic, hashable form.

    Pipeline (OSWorld-native filter + state-key linearize, then 3 steps):
      1. ``_semantic_filter_nodes`` — starts with OSWorld's visibility gate,
         then retains textual editable document descendants for offscreen edits.
      2. ``_linearize_accessibility_tree`` — emit OSWorld's stable columns
         {tag, name, text, class, description} (+ coords only if keep_coords);
         every volatile st:*/val:*/act:* attribute is already excluded by only
         emitting these columns.
      3. (anchor-specific) drop the position/size columns by default — they
         jitter on window move/resize.
      4. (anchor-specific) sort the flat row multiset — the server assembles
         top-level windows via ``concurrent.futures.as_completed``
         (server/main.py:916), so document order is non-deterministic; sorting
         makes two snapshots of the SAME desktop produce an identical string.
      Capping canonical string is then hashed by a11y_anchor_hash.

    ``keep_text`` only applies to non-EditWrapper nodes' text (vendor parity).
    Set it False to ignore typed-in text-field content (rarely needed; defaults
    to True so filled-vs-empty form fields stay distinguishable).

    Returns "" on parse failure / None so callers can fall back to the task
    instruction as the anchor (safe GiGPO degradation toward episode-only).
    """
    if not a11y_xml:
        return ""
    try:
        rows = _linearize_accessibility_tree(
            a11y_xml, platform=platform, keep_coords=keep_coords
        )
    except ET.ParseError:
        logger.warning("a11y anchor: failed to parse XML; returning empty key")
        return ""
    except ValueError as e:  # bad platform
        raise
    if not keep_text:
        # Blank the text column (index 2) to ignore dynamic/typed content.
        rows = [(r.split("\t", 2)[:2] + [""] + r.split("\t")[3:]) if "\t" in r else r
                for r in rows]
    rows.sort()
    return "\n".join(rows)


def a11y_anchor_hash(
    a11y_xml: str | None,
    *,
    platform: str = "ubuntu",
    keep_coords: bool = False,
    keep_text: bool = True,
) -> str:
    """SHA-256 of the normalized a11y tree. Empty string when the key is empty."""
    canonical = normalize_a11y(
        a11y_xml, keep_coords=keep_coords, keep_text=keep_text, platform=platform
    )
    if not canonical:
        return ""
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Semantic canonical anchor (hybrid exact + conflict-only grouping)           #
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u2060\ufeff]")
_TIME_RE = re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[ap]m)?\b", re.I)
_DATE_RE = re.compile(r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b", re.I)
_PERCENT_RE = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\s*%")
_VOLATILE_HINTS = ("clock", "time", "date", "progress", "status", "notification", "toast", "timer")
_CRITICAL_STATES = (
    "checked",
    "selected",
    "expanded",
    "pressed",
    "indeterminate",
    "editable",
    "enabled",
    "focused",
)
DEFAULT_NODE_DIFFERENCE_CONFLICT_THRESHOLD = 3


def _clean_text(value: str | None, *, normalize_dynamic: bool = False) -> str:
    """Normalize UI labels conservatively; dynamic patterns only on volatile widgets."""
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip().casefold()
    if not normalize_dynamic or not text:
        return text
    text = _UUID_RE.sub("<uuid>", text)
    text = _TIME_RE.sub("<time>", text)
    text = _DATE_RE.sub("<date>", text)

    def _percent_bucket(match: re.Match) -> str:
        value = float(match.group(1))
        if value <= 0:
            return "<percent:zero>"
        if value >= 100:
            return "<percent:done>"
        return "<percent:active>"

    return _PERCENT_RE.sub(_percent_bucket, text)


def _node_class(node: ET.Element, platform: str) -> str:
    if platform == "ubuntu":
        return node.get(f"{{{_attributes_ns_ubuntu}}}class", "")
    return node.get(f"{{{_class_ns_windows}}}class", "")


def _node_description(node: ET.Element, platform: str) -> str:
    ns = _attributes_ns_ubuntu if platform == "ubuntu" else _attributes_ns_windows
    return node.get(f"{{{ns}}}description", "")


def _nearest_parent_signature(
    node: ET.Element,
    parent_by_node: dict[ET.Element, ET.Element],
    platform: str,
) -> str:
    """Keep hierarchy without sibling indices, which are unstable across captures."""
    parent = parent_by_node.get(node)
    while parent is not None:
        name = _clean_text(parent.get("name", ""))
        cls = _clean_text(_node_class(parent, platform))
        if name or cls or parent.tag.startswith("document"):
            return "|".join((_clean_text(parent.tag), name, cls))
        parent = parent_by_node.get(parent)
    return ""


@dataclass(frozen=True)
class CanonicalA11yAnchor:
    """Canonical semantic state plus feature multisets used for conflict checks."""

    exact_key: str
    canonical: str
    structural_features: tuple[str, ...]
    full_features: tuple[str, ...]
    critical_by_structure: tuple[tuple[str, str], ...]
    content_by_structure: tuple[tuple[str, str], ...]
    editable_text_features: tuple[str, ...]
    dialog_features: frozenset[str]
    context_features: frozenset[str]
    node_count: int


def canonicalize_a11y(a11y_xml: str | None, *, platform: str = "ubuntu") -> CanonicalA11yAnchor | None:
    """Build a hierarchy-aware, semantic canonical representation.

    Unlike the legacy hash, this representation keeps decision-relevant widget
    states (checked/selected/expanded/value) and a stable parent signature. It
    still removes coordinates, sibling order, focus jitter, and known volatile
    clock/progress text.
    """
    if not a11y_xml:
        return None
    if platform not in {"ubuntu", "windows"}:
        raise ValueError("Invalid platform, must be 'ubuntu' or 'windows'")
    try:
        root = ET.fromstring(a11y_xml)
        nodes = _semantic_filter_nodes(root, platform)
    except ET.ParseError:
        logger.warning("semantic a11y anchor: failed to parse XML")
        return None

    parent_by_node = {child: parent for parent in root.iter() for child in parent}
    state_ns = _state_ns_ubuntu if platform == "ubuntu" else _state_ns_windows
    value_ns = _value_ns_ubuntu if platform == "ubuntu" else _value_ns_windows
    structural: list[str] = []
    full: list[str] = []
    critical: list[tuple[str, str]] = []
    content: list[tuple[str, str]] = []
    editable_text: list[str] = []
    dialogs: set[str] = set()
    contexts: set[str] = set()

    for node in root.iter():
        tag = _clean_text(node.tag)
        if tag not in {"dialog", "alert"}:
            continue
        visible = node.get(f"{{{state_ns}}}visible") == "true"
        showing = node.get(f"{{{state_ns}}}showing") == "true"
        if not visible or (platform == "ubuntu" and not showing):
            continue
        ancestor = parent_by_node.get(node)
        application = ""
        while ancestor is not None:
            if ancestor.tag == "application":
                application = _clean_text(ancestor.get("name"))
                break
            ancestor = parent_by_node.get(ancestor)
        dialogs.add(
            "\t".join(
                (
                    application,
                    tag,
                    _clean_text(node.get("name")),
                    _clean_text(_node_class(node, platform)),
                )
            )
        )

    for node in nodes:
        tag = _clean_text(node.tag)
        raw_name = node.get("name", "")
        raw_class = _node_class(node, platform)
        raw_desc = _node_description(node, platform)
        volatile = any(
            hint in " ".join((tag, raw_name.casefold(), raw_class.casefold(), raw_desc.casefold()))
            for hint in _VOLATILE_HINTS
        )
        name = _clean_text(raw_name, normalize_dynamic=volatile)
        cls = _clean_text(raw_class)
        desc = _clean_text(raw_desc, normalize_dynamic=volatile)
        text = _clean_text(node.text, normalize_dynamic=volatile)
        value = _clean_text(node.get(f"{{{value_ns}}}value", ""), normalize_dynamic=False)
        parent = _nearest_parent_signature(node, parent_by_node, platform)
        structure = "\t".join((parent, tag, name, cls, desc))
        state_parts = [
            f"{key}={_clean_text(node.get(f'{{{state_ns}}}{key}', ''))}"
            for key in _CRITICAL_STATES
            if node.get(f"{{{state_ns}}}{key}") is not None
        ]
        state = ";".join(state_parts)
        full_row = "\t".join((structure, f"text={text}", f"value={value}", f"state={state}"))
        # Scroll position is continuous viewport jitter, not a semantic widget
        # transition. Keep it in full_features, but omit its value from the
        # hard content-conflict signal.
        critical_state = state if tag in {"scroll-bar", "scrollbar"} else "\t".join(
            (f"value={value}", f"state={state}")
        )
        content_value = "" if tag in {"scroll-bar", "scrollbar"} else value
        content_state = "\t".join(
            (f"text={text}", f"value={content_value}", f"state={state}")
        )
        structural.append(structure)
        full.append(full_row)
        critical.append((structure, critical_state))
        content.append((structure, content_state))
        if node.get(f"{{{state_ns}}}editable") == "true" and text:
            ancestor = parent_by_node.get(node)
            in_document = tag.startswith("document")
            while ancestor is not None and not in_document:
                in_document = _clean_text(ancestor.tag).startswith("document")
                ancestor = parent_by_node.get(ancestor)
            if in_document:
                editable_text.append("\t".join((structure, f"text={text}")))
        if tag.startswith("document") or not parent:
            contexts.add("\t".join((tag, name, cls)))

    if not full:
        return None
    structural.sort()
    full.sort()
    critical.sort()
    content.sort()
    editable_text.sort()
    canonical = "\n".join(full)
    return CanonicalA11yAnchor(
        exact_key=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        canonical=canonical,
        structural_features=tuple(structural),
        full_features=tuple(full),
        critical_by_structure=tuple(critical),
        content_by_structure=tuple(content),
        editable_text_features=tuple(editable_text),
        dialog_features=frozenset(dialogs),
        context_features=frozenset(contexts),
        node_count=len(full),
    )


def _multiset_dice(left: Sequence[str], right: Sequence[str]) -> tuple[float, int]:
    if not left and not right:
        return 1.0, 0
    a, b = Counter(left), Counter(right)
    shared = sum((a & b).values())
    return (2.0 * shared / max(1, len(left) + len(right))), shared


def semantic_anchor_similarity(
    left: CanonicalA11yAnchor,
    right: CanonicalA11yAnchor,
) -> tuple[float, int, bool]:
    """Return diagnostic similarity, shared nodes, and conflict status.

    Similarity is retained for reports only. Conflict-only hybrid grouping does
    not compare this score with a threshold.
    """
    structural_score, full_score, shared, conflict = semantic_anchor_similarity_details(left, right)
    return min(structural_score, full_score), shared, conflict


def semantic_anchor_similarity_details(
    left: CanonicalA11yAnchor,
    right: CanonicalA11yAnchor,
) -> tuple[float, float, int, bool]:
    """Return structural score, full score, shared nodes, and state conflict."""
    structural_score, shared = _multiset_dice(left.structural_features, right.structural_features)
    full_score, _ = _multiset_dice(left.full_features, right.full_features)

    conflicts = semantic_anchor_conflict_details(left, right)
    return structural_score, full_score, shared, conflicts.any_conflict


@dataclass(frozen=True)
class SemanticAnchorConflictDetails:
    critical_state_conflict: bool
    editable_text_conflict: bool
    dialog_presence_conflict: bool
    context_disjoint: bool
    node_difference_count: int
    node_difference_conflict: bool

    @property
    def any_conflict(self) -> bool:
        return any((
            self.critical_state_conflict,
            self.editable_text_conflict,
            self.dialog_presence_conflict,
            self.context_disjoint,
            self.node_difference_conflict,
        ))


def semantic_anchor_conflict_details(
    left: CanonicalA11yAnchor,
    right: CanonicalA11yAnchor,
    *,
    node_difference_threshold: int = DEFAULT_NODE_DIFFERENCE_CONFLICT_THRESHOLD,
) -> SemanticAnchorConflictDetails:
    """Return every hard conflict used by hybrid state grouping."""
    critical_state_conflict, editable_text_conflict = semantic_anchor_conflicts(left, right)
    left_full = Counter(left.full_features)
    right_full = Counter(right.full_features)
    only_left = sum((left_full - right_full).values())
    only_right = sum((right_full - left_full).values())
    difference_count = max(only_left, only_right)
    threshold = max(1, int(node_difference_threshold))
    context_disjoint = bool(
        left.context_features
        and right.context_features
        and left.context_features.isdisjoint(right.context_features)
    )
    return SemanticAnchorConflictDetails(
        critical_state_conflict=critical_state_conflict,
        editable_text_conflict=editable_text_conflict,
        dialog_presence_conflict=left.dialog_features != right.dialog_features,
        context_disjoint=context_disjoint,
        node_difference_count=difference_count,
        node_difference_conflict=difference_count >= threshold,
    )


def semantic_anchor_conflicts(
    left: CanonicalA11yAnchor,
    right: CanonicalA11yAnchor,
) -> tuple[bool, bool]:
    """Return ``(content_conflict, editable_document_text_conflict)``.

    Content is compared as a multiset for every structure, so repeated
    anonymous controls (for example two spin-buttons) are not silently
    discarded by a uniqueness check.
    """
    left_map: dict[str, Counter[str]] = defaultdict(Counter)
    right_map: dict[str, Counter[str]] = defaultdict(Counter)
    for structure, content in left.content_by_structure:
        left_map[structure][content] += 1
    for structure, content in right.content_by_structure:
        right_map[structure][content] += 1
    content_conflict = any(
        left_map[key] != right_map[key]
        for key in left_map.keys() & right_map.keys()
    )
    editable_conflict = Counter(left.editable_text_features) != Counter(right.editable_text_features)
    return content_conflict, editable_conflict


# --------------------------------------------------------------------------- #
# Grouping helpers                                                            #
# --------------------------------------------------------------------------- #
def _resolve_anchor(sample: Any, *, platform: str = "ubuntu") -> str:
    """Return the grouping key for one step-sample.

    Prefers the normalized a11y state stored on ``metadata["gigpo"]``; if the
    rollout only saved the raw ``anchor_obs`` text, hash it here (the normalizer
    is numpy-free, so this is cheap). Finally falls back to the instruction
    string so an env without a11y still trains.
    """
    meta = getattr(sample, "metadata", None) or {}
    gigpo_meta = meta.get("gigpo") if isinstance(meta, dict) else None
    if isinstance(gigpo_meta, dict):
        anchor_hash = gigpo_meta.get("anchor_hash")
        if isinstance(anchor_hash, str) and anchor_hash:
            return anchor_hash
        obs = gigpo_meta.get("anchor_obs")
        if isinstance(obs, str) and obs:
            hashed = a11y_anchor_hash(obs, platform=platform)
            if hashed:
                return hashed
    # Exact mode may fall back to instruction. Hybrid mode instead keeps
    # missing states singleton via _resolve_semantic_anchor.
    instr = meta.get("instruction") if isinstance(meta, dict) else None
    if not instr and isinstance(gigpo_meta, dict):
        instr = gigpo_meta.get("instruction")
    return instr if isinstance(instr, str) and instr else ""


@dataclass(frozen=True)
class _ResolvedAnchor:
    key: str
    semantic: CanonicalA11yAnchor | None
    source: str


def _resolve_semantic_anchor(sample: Any, *, platform: str) -> _ResolvedAnchor:
    meta = getattr(sample, "metadata", None) or {}
    gigpo_meta = meta.get("gigpo") if isinstance(meta, dict) else None
    obs = gigpo_meta.get("anchor_obs") if isinstance(gigpo_meta, dict) else None
    if isinstance(obs, str) and obs:
        anchor = canonicalize_a11y(obs, platform=platform)
        if anchor is not None:
            return _ResolvedAnchor(anchor.exact_key, anchor, "a11y")

    # Missing/malformed a11y must stay a singleton. Grouping all missing states
    # by task instruction invents a false step comparison signal.
    gi = getattr(sample, "group_index", -1)
    ti = getattr(sample, "index", -1)
    step = gigpo_meta.get("step_index", -1) if isinstance(gigpo_meta, dict) else -1
    return _ResolvedAnchor(f"missing:{gi}:{ti}:{step}:{id(sample)}", None, "missing")


def build_step_group(
    group_index_by_sample: Sequence,
    anchor_keys: Sequence[str],
) -> List[str]:
    """Assign exact anchor groups within each episode group."""
    step_group_uids: List[str] = [""] * len(anchor_keys)
    seen_groups = set(group_index_by_sample)
    for gidx in seen_groups:
        locs = [i for i, gi in enumerate(group_index_by_sample) if gi == gidx]
        if not locs:
            continue
        clusters: Dict[str, List[int]] = defaultdict(list)
        for i in locs:
            clusters[anchor_keys[i]].append(i)
        for members in clusters.values():
            uid = str(uuid.uuid4())
            for i in members:
                step_group_uids[i] = uid
    return step_group_uids


def build_hybrid_step_group(
    group_index_by_sample: Sequence,
    anchors: Sequence[_ResolvedAnchor],
    *,
    step_indices: Sequence[int] | None = None,
    node_difference_conflict_threshold: int = DEFAULT_NODE_DIFFERENCE_CONFLICT_THRESHOLD,
) -> tuple[List[str], Dict[str, Any]]:
    """Exact buckets, then chronological root-only conflict clustering."""
    if len(group_index_by_sample) != len(anchors):
        raise ValueError("group indices and anchors must have equal length")
    if step_indices is None:
        step_indices = list(range(len(anchors)))
    if len(step_indices) != len(anchors):
        raise ValueError("step indices and anchors must have equal length")

    uids = [""] * len(anchors)
    match_kind = ["singleton"] * len(anchors)
    fuzzy_bucket_merges = 0
    exact_bucket_merges = 0

    for group_index in set(group_index_by_sample):
        indices = [i for i, value in enumerate(group_index_by_sample) if value == group_index]
        exact_buckets: dict[str, list[int]] = defaultdict(list)
        for i in indices:
            exact_buckets[anchors[i].key].append(i)
        exact_bucket_merges += sum(max(0, len(members) - 1) for members in exact_buckets.values())

        # Earliest states become stable roots. Later states compare only with
        # roots, matching rollout semantics where a repeated state is commonly
        # reached by returning to an earlier state with harmless residual jitter.
        clusters: list[dict[str, Any]] = []
        ordered_keys = sorted(
            exact_buckets,
            key=lambda key: (
                min(int(step_indices[i]) for i in exact_buckets[key]),
                min(exact_buckets[key]),
            ),
        )
        for key in ordered_keys:
            members = exact_buckets[key]
            representative_index = min(
                members, key=lambda i: (int(step_indices[i]), i)
            )
            representative = anchors[representative_index]
            placed = False
            if representative.semantic is not None:
                for cluster in clusters:
                    root_anchor = cluster["root"]
                    if root_anchor.semantic is None:
                        continue
                    conflicts = semantic_anchor_conflict_details(
                        representative.semantic,
                        root_anchor.semantic,
                        node_difference_threshold=node_difference_conflict_threshold,
                    )
                    if not conflicts.any_conflict:
                        cluster["members"].extend(members)
                        cluster["has_fuzzy_merge"] = True
                        fuzzy_bucket_merges += 1
                        placed = True
                        break
            if not placed:
                clusters.append(
                    {
                        "members": list(members),
                        "root": representative,
                        "has_fuzzy_merge": False,
                    }
                )

        for cluster in clusters:
            uid = str(uuid.uuid4())
            members = cluster["members"]
            if len(members) > 1:
                kind = "fuzzy" if cluster["has_fuzzy_merge"] else "exact"
            else:
                kind = "singleton"
            for i in members:
                uids[i] = uid
                match_kind[i] = kind

    group_sizes = Counter(uids)
    diagnostics = {
        "exact_sample_merges": exact_bucket_merges,
        "fuzzy_bucket_merges": fuzzy_bucket_merges,
        "singleton_count": sum(1 for uid in uids if group_sizes[uid] == 1),
        "match_kind": match_kind,
        "grouping_policy": "conflict_only",
        "clustering_policy": "chronological_root",
        "node_difference_conflict_threshold": max(
            1, int(node_difference_conflict_threshold)
        ),
    }
    return uids, diagnostics


# --------------------------------------------------------------------------- #
# Discounted returns + advantage normalization (ported from core_gigpo)       #
# --------------------------------------------------------------------------- #

def _norm_group(
    scores: List[float],
    *,
    remove_std: bool,
    epsilon: float,
) -> List[float]:
    """Normalize a state group, assigning zero advantage to singletons.

    Mirrors core_gigpo's convention (``mode="mean_norm"`` ⇒ ``remove_std=True``
    ⇒ mean-subtraction only, i.e. the std is *removed* from the normalization;
    ``mode="mean_std_norm"`` ⇒ ``remove_std=False`` ⇒ also divide by std). A
    one-element group has no peer action to compare against, so its state-level
    advantage is zero in both modes.
    """
    if len(scores) == 1:
        return [0.0]
    arr = np.asarray(scores, dtype=np.float64)
    mean = float(arr.mean())
    if remove_std:
        # Mean-normalize without dividing by std (ClawGUI "mean_norm").
        return [float(arr[i] - mean) for i in range(len(arr))]
    std = float(arr.std())
    return [float(arr[i] - mean) / (std + epsilon) for i in range(len(arr))]


def _trajectory_discounted_returns(
    step_rewards_by_pos: List[float], gamma: float
) -> List[float]:
    """``R_t = r_t + γ·R_{t+1}`` backward within one trajectory (Eq. 5)."""
    n = len(step_rewards_by_pos)
    returns = [0.0] * n
    running = 0.0
    for t in range(n - 1, -1, -1):
        running = float(step_rewards_by_pos[t]) + gamma * running
        returns[t] = running
    return returns


def _trajectory_group_advantages(
    trajectory_keys: Sequence[Tuple[int, int]],
    outcomes: Dict[Tuple[int, int], float],
    *,
    std_normalization: bool,
    epsilon: float,
) -> Dict[Tuple[int, int], float]:
    """Compute GRPO-style outcome advantages once per trajectory.

    Dynamic-history samples repeat a trajectory outcome at every action step.
    We de-duplicate by ``(group_index, trajectory_index)`` before calculating
    the prompt baseline, matching Slime's dynamic GRPO reward path. A prompt
    with only one trajectory has no relative trajectory signal, so its outcome
    advantage is zero; the state-level PRM term can still train that sample.
    """
    group_to_keys: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for key in trajectory_keys:
        if key not in group_to_keys[key[0]]:
            group_to_keys[key[0]].append(key)

    advantages: Dict[Tuple[int, int], float] = {}
    for keys in group_to_keys.values():
        values = np.asarray([outcomes[key] for key in keys], dtype=np.float64)
        centered = values - float(values.mean())
        if std_normalization and len(values) > 1:
            centered = centered / (float(values.std(ddof=1)) + epsilon)
        for key, value in zip(keys, centered, strict=True):
            advantages[key] = float(value)
    return advantages


def compute_gigpo_advantages(
    samples: List[Any],
    *,
    step_advantage_w: float = 1.0,
    gamma: float = 0.95,
    mode: str = "mean_norm",
    anchor_mode: str | None = None,
    anchor_node_difference_conflict_threshold: int = DEFAULT_NODE_DIFFERENCE_CONFLICT_THRESHOLD,
    platform: str = "ubuntu",
    epsilon: float = 1e-6,
    trajectory_std_normalization: bool = True,
) -> Tuple[List[float], Dict[str, Any]]:
    """Compute separate trajectory and state-relative step advantages.

    Args:
        samples: per-step training samples for ONE prompt group (all share
            ``group_index``; trajectories within differ by ``index``). Each must
            carry ``metadata["gigpo"]`` (anchor) and ``metadata["step_wise"]``
            (step_scores / outcome_reward).
        step_advantage_w: Weight ``w`` on the state-relative analyzer advantage.
            The legacy name is retained for training-script compatibility.
        gamma: discount for within-trajectory step returns.
        mode: ``"mean_norm"`` (subtract group mean) or ``"mean_std_norm"``
            (also divide by group std).
        anchor_mode: ``exact`` or conflict-only ``hybrid``. When None, uses
            exact grouping for backward compatibility.
        anchor_node_difference_conflict_threshold: reject a hybrid merge when
            at least this many canonical full nodes differ on either side.
        epsilon: numerical stabilizer for std division.
        trajectory_std_normalization: Divide trajectory advantages by their
            sample standard deviation after prompt-level mean subtraction.
    Returns:
        ``(advantages, diagnostics)`` where each scalar is the unscaled
        ``A_traj + step_advantage_w * A_step``. Dynamic-history trajectory
        scaling is deferred until Slime has the complete rollout batch and can
        compute a single batch-wide ``mean(T)``.
    """
    if mode not in ("mean_norm", "mean_std_norm"):
        raise ValueError(f"Unknown gigpo mode: {mode}")
    if anchor_mode is None:
        anchor_mode = "exact"
    anchor_mode = str(anchor_mode).strip().lower()
    if anchor_mode not in {"exact", "hybrid"}:
        raise ValueError(f"Unknown GiGPO anchor mode: {anchor_mode}")
    remove_std = mode == "mean_norm"
    n = len(samples)
    if n == 0:
        return [], {"num_samples": 0}

    # --- per-sample primitives ------------------------------------------------ #
    group_index = []
    traj_index = []
    anchor_keys: List[str] = []
    resolved_anchors: List[_ResolvedAnchor] = []
    anchor_sources: List[str] = []
    step_scores: List[float] = []
    step_pos: List[int] = []          # position within its trajectory (for γ)
    outcomes_seen: Dict[Tuple, float] = {}

    for i, s in enumerate(samples):
        meta = getattr(s, "metadata", None) or {}
        if isinstance(meta, dict) and meta.get("remove_sample"):
            # Should not normally be requested in gigpo mode, but be safe.
            pass
        gi = int(getattr(s, "group_index", -1) if getattr(s, "group_index", None) is not None else -1)
        ti = int(getattr(s, "index", i) if getattr(s, "index", None) is not None else i)
        group_index.append(gi)
        traj_index.append((gi, ti))

        if anchor_mode == "hybrid":
            resolved = _resolve_semantic_anchor(s, platform=platform)
            resolved_anchors.append(resolved)
            anchor_keys.append(resolved.key)
            anchor_sources.append(resolved.source)
        else:
            anchor_keys.append(_resolve_anchor(s, platform=platform))
            anchor_sources.append("a11y" if anchor_keys[-1] else "missing")

        gigpo_meta = meta.get("gigpo") if isinstance(meta, dict) else None
        step_pos.append(int(gigpo_meta.get("step_index", i)) if isinstance(gigpo_meta, dict) else i)

        step_wise = meta.get("step_wise") if isinstance(meta, dict) else None
        step_score = 0.0
        if isinstance(step_wise, dict):
            raw = step_wise.get("step_scores", [])
            if isinstance(raw, list) and raw:
                try:
                    step_score = float(raw[0])
                except (TypeError, ValueError):
                    step_score = 0.0
            outcome = step_wise.get("outcome_reward")
            if outcome is None:
                outcome = step_wise.get("dynamic_outcome_reward")
        else:
            outcome = None
        step_scores.append(step_score)

        # Prefer the trajectory outcome recorded on metadata; dedupe so a
        # trajectory's many step-samples do not get counted against each other.
        if outcome is None:
            reward = getattr(s, "reward", None)
            outcome = float(reward.get("score", 0.0)) if isinstance(reward, dict) else float(reward or 0.0)
        outcomes_seen[(gi, ti)] = float(outcome)

    # Keep the task/episode count for diagnostics. Outcomes are normalized once
    # per trajectory inside their prompt group, before dynamic-history scaling.
    group_to_trajs: Dict[int, List[Tuple]] = defaultdict(list)
    for key in traj_index:
        group_to_trajs[key[0]].append(key)

    trajectory_advantages = _trajectory_group_advantages(
        traj_index,
        outcomes_seen,
        std_normalization=trajectory_std_normalization,
        epsilon=epsilon,
    )
    unscaled_trajectory_advantages = [trajectory_advantages[key] for key in traj_index]

    # --- discounted per-step returns within each trajectory (Eq. 5) ---------- #
    returns = [0.0] * n
    traj_to_positions: Dict[Tuple, List[int]] = defaultdict(list)
    for i in range(n):
        traj_to_positions[traj_index[i]].append(i)
    for key, idxs in traj_to_positions.items():
        idxs_sorted = sorted(idxs, key=lambda i: step_pos[i])
        r_seq = [step_scores[i] for i in idxs_sorted]
        disc = _trajectory_discounted_returns(r_seq, gamma=gamma)
        for j, i in enumerate(idxs_sorted):
            returns[i] = disc[j]

    # --- state-level relative advantage via anchor micro-groups --------------- #
    hybrid_diag: Dict[str, Any] = {
        "exact_sample_merges": 0,
        "fuzzy_bucket_merges": 0,
        "singleton_count": 0,
        "match_kind": ["legacy"] * n,
    }
    if anchor_mode == "hybrid":
        step_group_uids, hybrid_diag = build_hybrid_step_group(
            group_index,
            resolved_anchors,
            step_indices=step_pos,
            node_difference_conflict_threshold=anchor_node_difference_conflict_threshold,
        )
    else:
        step_group_uids = build_step_group(group_index, anchor_keys)
    uid_to_returns: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for i in range(n):
        uid_to_returns[step_group_uids[i]].append((i, returns[i]))
    step_advantages = [0.0] * n
    group_sizes: List[int] = []
    for uid, members in uid_to_returns.items():
        rewards = [m[1] for m in members]
        group_sizes.append(len(members))
        norm = _norm_group(
            rewards,
            remove_std=remove_std,
            epsilon=epsilon,
        )
        for (i, _), val in zip(members, norm):
            step_advantages[i] = val

    advantages = [
        unscaled_trajectory_advantages[i] + step_advantage_w * step_advantages[i]
        for i in range(n)
    ]

    uid_sizes = Counter(step_group_uids)
    for i, sample in enumerate(samples):
        meta = getattr(sample, "metadata", None)
        if not isinstance(meta, dict):
            continue
        gigpo_meta = meta.get("gigpo")
        if not isinstance(gigpo_meta, dict):
            gigpo_meta = {}
        if anchor_mode == "hybrid" and resolved_anchors[i].semantic is not None:
            gigpo_meta["anchor_hash"] = resolved_anchors[i].key
            gigpo_meta["anchor_node_count"] = resolved_anchors[i].semantic.node_count
        gigpo_meta["gigpo_diag"] = {
            "trajectory_reward": float(outcomes_seen[traj_index[i]]),
            "trajectory_advantage": float(trajectory_advantages[traj_index[i]]),
            # Slime overwrites this after computing batch-wide mean(T).
            "trajectory_advantage_scaled": float(unscaled_trajectory_advantages[i]),
            "discounted_step_return": float(returns[i]),
            "step_advantage": float(step_advantages[i]),
            "weighted_step_advantage": float(step_advantage_w * step_advantages[i]),
            "advantage": float(advantages[i]),
            "anchor_key": anchor_keys[i],
            "anchor_source": anchor_sources[i],
            "anchor_mode": anchor_mode,
            "anchor_match_kind": hybrid_diag["match_kind"][i],
            "step_group_uid": step_group_uids[i],
            "step_group_size": int(uid_sizes[step_group_uids[i]]),
            "singleton_step_advantage_zeroed": uid_sizes[step_group_uids[i]] == 1,
            "group_index": group_index[i],
            "trajectory_index": traj_index[i],
        }
        meta["gigpo"] = gigpo_meta

    diagnostics = {
        "num_samples": n,
        "num_episodes": len(group_to_trajs),
        "num_step_groups": len(uid_to_returns),
        "avg_step_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "mode": mode,
        "step_advantage_w": step_advantage_w,
        "trajectory_std_normalization": trajectory_std_normalization,
        "trajectory_advantage_scaling": "deferred_to_rollout_batch",
        "gamma": gamma,
        "anchor_mode": anchor_mode,
        "empty_anchor_count": sum(1 for source in anchor_sources if source == "missing"),
        "exact_sample_merges": hybrid_diag["exact_sample_merges"],
        "fuzzy_bucket_merges": hybrid_diag["fuzzy_bucket_merges"],
        "singleton_count": (
            hybrid_diag["singleton_count"]
            if anchor_mode == "hybrid"
            else sum(1 for uid in step_group_uids if uid_sizes[uid] == 1)
        ),
        "singleton_zeroed_count": sum(1 for uid in step_group_uids if uid_sizes[uid] == 1),
    }
    if diagnostics["empty_anchor_count"]:
        logger.warning(
            "gigpo: %d/%d samples have empty anchor (a11y unavailable?) — "
            "hybrid mode keeps them as singletons",
            diagnostics["empty_anchor_count"], n,
        )
    return advantages, diagnostics


__all__ = [
    "normalize_a11y",
    "a11y_anchor_hash",
    "CanonicalA11yAnchor",
    "SemanticAnchorConflictDetails",
    "DEFAULT_NODE_DIFFERENCE_CONFLICT_THRESHOLD",
    "canonicalize_a11y",
    "semantic_anchor_similarity",
    "semantic_anchor_similarity_details",
    "semantic_anchor_conflicts",
    "semantic_anchor_conflict_details",
    "build_step_group",
    "build_hybrid_step_group",
    "compute_gigpo_advantages",
]
