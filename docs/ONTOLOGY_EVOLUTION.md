# xCH Ontology Evolution Guide

This guide explains how to extend the xch-MIND system when the analysis of new data requires introducing new concepts, properties, or relations into the `xch:` (xch-MIND) interpretive ontology.

## Why evolve the ontology?

The system is designed to be rigorous. If an agent extracts a new type of information (e.g., altitude or construction materials) but the system isn't instructed to handle it, the information will be lost or flagged as invalid by the `TripleValidator`.

---

## Procedure for Adding a New Property

If you want the system to extract and validate a new property (e.g., `xch:hasMaterial`), follow these 5 steps:

### 1. Define in the Data Model (Python)

Update the Pydantic model for the relevant agent in `src/agents/models.py` (or specific worker files).

- **File**: [models.py](src/agents/models.py)
- **Action**: Add the new field to the agent's response class (e.g., `TypologicalAnalysisResponse`).

### 2. Update Agent Logic

Modify the agent's prompt and logic to start extracting that data.

- **File**: `src/agents/workers/[agent_name].py`
- **Action**: Update the system prompt and the `analyze` function to populate the new model field.

### 3. Map to RDF (TripleGenerator)

Instruct the generator on how to transform that JSON field into an RDF triple.

- **File**: [generator.py](src/triples/generator.py)
- **Action**: Add logic to the `_add_provenance` function or specific methods (e.g., `_add_typological_cluster`) to add the triple to the graph.
  ```python
  if assertion.get("material"):
      graph.add((subject, XCH.hasMaterial, Literal(assertion["material"])))
  ```

### 4. Whitelist in the Validator (TripleValidator)

If you don't add the property here, the system will generate an "Unknown property" Warning.

- **File**: [validator.py](src/triples/validator.py)
- **Action**: Add `XCH.propertyName` to the `XCH_PROPERTIES` set.

### 5. Update Tests

Ensure the new property is correctly generated and validated.

- **File**: `tests/test_triple_generator.py`
- **Action**: Add a test case that includes the new property and verifies its presence in the final graph.

---

## Summary Table: What and Where

| Objective                   | File to Modify             | Component                 |
| :-------------------------- | :------------------------- | :------------------------ |
| **New extracted data type** | `src/agents/models.py`     | JSON Schema / Pydantic    |
| **New extraction logic**    | `src/agents/workers/*.py`  | LLM Prompt / Worker Logic |
| **New RDF predicate**       | `src/triples/generator.py` | JSON -> RDF Mapping       |
| **New validity rule**       | `src/triples/validator.py` | Ontology (Whitelist)      |
| **System consistency**      | `tests/test_triple_*.py`   | Test Suite                |

> [!TIP]
> Always keep the Python field name and the ontology property name synchronized to avoid confusion during debugging.

---

## Future Work

### Template-based Triple Generator

The current `TripleGenerator` uses hardcoded Python methods for each assertion type (one `_add_*` method per `assertion_type`). This means adding new triple patterns requires modifying Python source code.

A **template-based approach** would externalise the assertion→triple mapping into external template files (e.g., SPARQL CONSTRUCT or Jinja2 templates stored in `ontology/xch/templates/`). Each `assertion_type` would have a corresponding `.sparql.j2` template populated at runtime with assertion fields.

**Benefits:**

- Adding new triple patterns requires editing a template file, not Python code
- Domain experts can review and modify RDF generation without touching the pipeline
- Templates can be versioned independently from the codebase
- Enables A/B testing of different triple structures

**Estimated effort:** ~2 days for the refactoring, low risk (same logic, just externalised).

> [!NOTE]
> This is a refactoring improvement, not a correctness fix. The current deterministic mapping guarantees valid output and is appropriate for the current research scope.
