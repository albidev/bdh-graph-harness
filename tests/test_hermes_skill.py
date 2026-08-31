from pathlib import Path
import re

import yaml


ROOT = Path(__file__).parents[1]
SKILL = ROOT / "skills" / "research" / "bdh-graph-harness" / "SKILL.md"
README = ROOT / "README.md"
LEGACY = ROOT / "docs" / "hermes-skill.md"


REQUIRED_SECTIONS = [
    "When to Use",
    "Prerequisites",
    "How to Run",
    "Quick Reference",
    "Procedure",
    "Pitfalls",
    "Verification",
]


def _frontmatter_and_body(content: str):
    match = re.match(r"\A---\n(.*?)\n---\n(.*)\Z", content, re.DOTALL)
    assert match, "skill must start with closed YAML frontmatter"
    return yaml.safe_load(match.group(1)), match.group(2)


def test_bdh_skill_has_standards_compliant_frontmatter():
    metadata, body = _frontmatter_and_body(SKILL.read_text(encoding="utf-8"))

    assert metadata["name"] == "bdh-graph-harness"
    assert len(metadata["description"]) <= 60
    assert metadata["description"].endswith(".")
    assert metadata["author"]
    assert metadata["license"]
    assert set(metadata["platforms"]) == {"linux", "macos", "windows"}
    assert metadata["metadata"]["hermes"]["tags"]
    assert body.strip()


def test_bdh_skill_uses_required_section_order():
    body = _frontmatter_and_body(SKILL.read_text(encoding="utf-8"))[1]
    headings = re.findall(r"^## (.+)$", body, re.MULTILINE)

    assert headings[: len(REQUIRED_SECTIONS)] == REQUIRED_SECTIONS


def test_bdh_skill_documents_native_tools_and_state_boundaries():
    content = SKILL.read_text(encoding="utf-8")

    assert "bdh_query" in content
    assert "bdh_stats" in content
    assert "Read-only" in content
    assert "Mutates" in content
    assert "bdh-hermes-bridge" in content
    assert "never assume" in content.lower()
    assert "<BDH_BASE_URL>" in content
    assert "localhost:8643" not in content


def test_readme_installs_canonical_skill_without_legacy_copy():
    readme = README.read_text(encoding="utf-8")
    legacy = LEGACY.read_text(encoding="utf-8")
    canonical_url = (
        "https://raw.githubusercontent.com/albidev/bdh-graph-harness/main/"
        "skills/research/bdh-graph-harness/SKILL.md"
    )

    assert canonical_url in readme
    assert "docs/hermes-skill.md` — copy" not in readme
    assert "canonical installable skill" in legacy
    assert len(legacy) < 1000
