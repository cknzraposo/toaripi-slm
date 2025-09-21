# Feature Specification: Offline-Capable Toaripi Educational Small Language Model (SLM)

**Feature Branch**: `001-offline-capable-toaripi`  
**Created**: 2025-09-18  
**Status**: Draft  
**Input**: User description: "Offline-capable Toaripi educational SLM spec for content generation (stories, vocab, Q&A, dialogues) with size <=7B params and reproducible pipeline"

## Execution Flow (main)

```text
1. Parse user description from Input
   → If empty: ERROR "No feature description provided"
2. Extract key concepts from description
   → Identify: actors (teachers, primary learners, content contributors), actions (generate stories, vocabulary lists, Q&A, dialogues), data (parallel English↔Toaripi corpus, prompts), constraints (≤7B params, offline, educational-only, reproducible pipeline)
3. For each unclear aspect:
   → (Resolved) All ambiguities assigned explicit defaults in this draft
4. Fill User Scenarios & Testing section
   → If no clear user flow: ERROR "Cannot determine user scenarios"
5. Generate Functional Requirements
   → Each requirement must be testable
6. Identify Key Entities (data involved)
7. Run Review Checklist
   → If implementation details found: ERROR "Remove tech details"
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines

- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no model architecture specifics, libraries, code)
- 👥 Written for educational and cultural stakeholders

### Section Requirements

- Mandatory sections completed
- Optional sections only if relevant
- Remove non-applicable sections

### For AI Generation

1. Ambiguities replaced with explicit decisions
2. Performance, retention, safety thresholds defined at baseline
3. Educational goals expressed in measurable form where possible
4. Future refinement can tighten metrics after pilot testing

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story

A primary school teacher wants culturally appropriate Toaripi reading material. They provide a short topic or seed idea and receive a simple Toaripi story suitable for young learners that reinforces language skills and basic knowledge themes. The system works fully offline in a classroom with no internet connection.

### Acceptance Scenarios

1. Given the teacher provides a topic (e.g., "sharing food"), When they request a story, Then the system returns a Toaripi story of short length (4–6 sentences) using age-appropriate vocabulary and no disallowed themes within 10 seconds on baseline CPU hardware.
2. Given the teacher requests vocabulary for a topic (e.g., "river life"), When the request is submitted, Then the system returns a list of 10 Toaripi words with English glosses and one short, simple Toaripi example sentence per word (≤12 words) within 8 seconds.
3. Given a learner-focused session offline, When a Q&A generation request is made based on a provided short text (≤120 words English or Toaripi), Then the system returns 3 comprehension question/answer pairs in Toaripi (answers ≤15 words) within 10 seconds.
4. Given a prompt for a dialogue scenario (e.g., "children preparing for a storm"), When requested, Then the system returns a multi-turn dialogue (6–8 turns) with at least two distinct speakers and no disallowed themes within 10 seconds.

### Edge Cases

- Empty prompt: System rejects with message "Please provide a topic or seed idea" and suggests 3 generic safe starter topics (e.g., sharing, weather, helping elders).
- Restricted topic terms detected: System blocks generation and returns a safe alternative suggestion plus explanation label "Restricted topic".
- Requested length exceeding limits: System caps to maximum per content type (story 8 sentences, vocabulary 20 items, dialogue 10 turns, Q&A 5 pairs) and informs user of cap.
- Parallel corpus below minimum threshold: Training refused; system reports "Need at least 150 aligned pairs; currently CURRENT_COUNT." (Minimum set to 150 to ensure basic linguistic coverage.)
- Duplicate prompt submitted within last 5 minutes: System returns cached prior output unless user explicitly selects "Regenerate".

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users (teachers) to request generation of four educational content types: stories, vocabulary lists, Q&A (comprehension), and dialogues.
- **FR-002**: System MUST operate fully offline after initial installation (no external network calls during generation or review).
- **FR-003**: System MUST constrain model size to ≤7B parameters pre-quantization and ≤5GB quantized footprint to enable deployment on baseline hardware (8GB RAM CPU-only).
- **FR-004**: System MUST produce Toaripi outputs with simple sentence structures (≤15 words per sentence average) and exclude disallowed themes.
- **FR-005**: System MUST block generation containing theological exposition, violence, adult themes, or cultural misappropriation using a rule-based lexical screen plus topic classification fallback.
- **FR-006**: System MUST accept a teacher-provided topic (2–80 characters) or seed text (≤300 characters) for each content type.
- **FR-007**: System MUST output vocabulary lists of a default 10 items (configurable 5–20) each with: Toaripi term, English gloss, one example sentence (≤12 words) in Toaripi.
- **FR-008**: System MUST validate input (non-empty, length bounds, characters limited to Latin + Toaripi orthography + basic punctuation) and return a structured error on failure.
- **FR-009**: System MUST store generation metadata (timestamp ISO8601, content type, token length, latency ms, safety status) locally and retain for 30 days before purge.
- **FR-010**: System MUST enable reproducibility via immutable version identifiers: data checksum, config version, and model version recorded with each generation.
- **FR-011**: System MUST provide a "stable" mode where identical prompt + type + seed produce similarity ≥90% (measured by token overlap) by enforcing fixed random seed.
- **FR-012**: System MUST block output if safety screening flags more than 0 high-risk terms or >3 medium-risk terms; user sees warning and no unsafe text.
- **FR-013**: System MUST support evaluation packs: on demand it generates a fixed 12-item sample set (3 per content type) for human review prior to release tagging.
- **FR-014**: System MUST refuse training if aligned corpus count <150 pairs; message instructs user to add more data.
- **FR-015**: System MUST allow specifying target lengths per type within constraints: story (3–8 sentences; default 5), vocabulary (5–20; default 10), Q&A pairs (2–5; default 3), dialogue turns (4–10; default 7).
- **FR-016**: System MUST provide a mechanism to output both Toaripi and optional English support for teachers (glosses / bilingual display) for all content types except dialogues (which remain Toaripi-only by default); default ON for vocabulary, OFF for others.
- **FR-017**: System MUST display a persistent disclaimer: "Educational prototype – Toaripi language preservation focus – not for theological or adult content." on each session start.
- **FR-018**: System MUST reject inputs exceeding 300 characters or containing unsupported symbols (anything outside allowed Unicode set) with a validation error.
- **FR-019**: System MUST ensure dialogues include at least two speakers alternating; no speaker may have two consecutive turns more than once.
- **FR-020**: System MUST complete generation within 10 seconds (stories, Q&A, dialogue) or 8 seconds (vocabulary list) on reference CPU (baseline: 8GB RAM single-thread fallback) in 95% of attempts.
- **FR-021**: System MUST cache last result per (content type, normalized topic) for 5 minutes and return cached version unless user chooses regenerate.
- **FR-022**: System MUST purge cached entries after 5 minutes or when storage exceeds 50MB.
- **FR-023**: System MUST provide a structured success response containing: content_type, content_body, metadata, safety_status, disclaimer_shown flag.
- **FR-024**: System MUST provide a structured error response with: error_code, message, remediation_hint.

### Key Entities

- **Content Request**: Attributes: id, topic, content_type, seed_text(optional), target_length, mode(stable|standard), include_english_support(bool), requested_at.
- **Generated Content Artifact**: Attributes: id, request_id, body, content_type, sentences_or_items_count, latency_ms, safety_status(pass|blocked|warning), model_version, data_checksum, config_version, created_at.
- **Parallel Corpus Entry**: Attributes: id, english_text, toaripi_text, source, checksum, approved(bool), added_at.
- **Safety Rule**: Attributes: id, category, pattern_or_term, severity(medium|high), active(bool), created_at.
- **Evaluation Pack**: Attributes: id, generated_at, model_version, sample_ids(list of Generated Content Artifact ids), reviewer_status(pending|approved|changes_requested).

---

## Review & Acceptance Checklist

GATE: Automated checks run during main() execution

### Content Quality

- [ ] No implementation details (languages, frameworks, APIs) present
- [ ] Focused on user value and educational outcomes
- [ ] Written for non-technical stakeholders
- [ ] All mandatory sections completed

### Requirement Completeness

- [ ] All requirements are testable and unambiguous
- [ ] Success criteria are measurable
- [ ] Scope is clearly bounded
- [ ] Dependencies and assumptions identified
- [ ] Performance, safety, and size constraints explicitly stated

---

## Execution Status

Updated by main() during processing

- [ ] User description parsed
- [ ] Key concepts extracted
- [ ] Ambiguities resolved
- [ ] User scenarios defined
- [ ] Requirements generated
- [ ] Entities identified
- [ ] Review checklist passed

---
