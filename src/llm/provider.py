"""
LLM Provider abstraction layer.

This module provides a unified interface for interacting with different LLM providers:
- Google Gemini (via langchain-google-genai)
- Ollama (local open-source models)

The provider is configurable via config.yaml and can be switched at runtime.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# BASE PROVIDER INTERFACE
# =============================================================================


class DailyQuotaExhaustedError(Exception):
    """
    Raised when the daily API quota is exhausted.

    This error should NOT be retried - it requires waiting until the next day
    or upgrading to a paid plan.
    """

    def __init__(
        self,
        message: str = "Daily API quota exhausted. Please wait until tomorrow or upgrade your plan.",
    ):
        super().__init__(message)
        self.message = message


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, rate_limiter: RateLimiter | None = None):
        """
        Initialize the provider.

        Args:
            rate_limiter: Optional RateLimiter for tracking and throttling
        """
        self.rate_limiter = rate_limiter

    @abstractmethod
    def get_model(self) -> BaseChatModel:
        """Return the underlying LangChain chat model."""
        pass

    @abstractmethod
    def invoke(self, messages: list[BaseMessage], **kwargs) -> str:
        """Invoke the model with messages and return the response text."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the model name for logging/debugging."""
        pass

    def create_prompt(self, system: str, human: str) -> ChatPromptTemplate:
        """Create a chat prompt template."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", human),
            ]
        )

    def _log_response(self, response: Any):
        """Log the LLM response in a human-readable format."""
        from pydantic import BaseModel
        import json

        def format_value(item: Any, indent: int = 0) -> str:
            tab = "  " * indent
            if isinstance(item, dict):
                lines = []
                for k, v in item.items():
                    key = f"\033[1;33m{k}\033[0m"  # Bold Yellow
                    if isinstance(v, (dict, list)) and v:
                        lines.append(f"{tab}{key}:")
                        lines.append(format_value(v, indent + 1))
                    else:
                        lines.append(f"{tab}{key}: {v}")
                return "\n".join(lines)
            elif isinstance(item, list):
                if not item:
                    return f"{tab}(empty)"
                lines = []
                for val in item:
                    bullet = f"\033[1;32m•\033[0m"  # Bold Green bullet
                    if isinstance(val, (dict, list)):
                        lines.append(f"{tab}{bullet}")
                        lines.append(format_value(val, indent + 1))
                    else:
                        lines.append(f"{tab}{bullet} {val}")
                return "\n".join(lines)
            else:
                return f"{tab}{item}"

        # Extract data
        data = response
        if isinstance(response, BaseModel):
            data = response.model_dump()
        elif hasattr(response, "content"):
            content = response.content
            # Try to parse string content as JSON if possible
            if isinstance(content, str) and content.strip().startswith(("{", "[")):
                try:
                    data = json.loads(content)
                except:
                    data = content
            else:
                data = content

        formatted = format_value(data)

        # Draw a box for the response
        separator = "─" * 40
        logger.info(f"LLM Response:\n{separator}\n{formatted}\n{separator}")

    def invoke_with_prompt(
        self,
        system_prompt: str,
        human_prompt: str,
        variables: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """
        Invoke the model with system and human prompts.

        Args:
            system_prompt: System message template
            human_prompt: Human message template
            variables: Variables to format into the prompts
            **kwargs: Additional arguments for the model

        Returns:
            Model response as string
        """
        variables = variables or {}

        # Format prompts
        system_formatted = system_prompt.format(**variables) if variables else system_prompt
        human_formatted = human_prompt.format(**variables) if variables else human_prompt

        messages = [
            SystemMessage(content=system_formatted),
            HumanMessage(content=human_formatted),
        ]

        return self.invoke(messages, **kwargs)

    def invoke_structured(
        self,
        system_prompt: str,
        human_prompt: str,
        output_schema: type,
        variables: dict[str, Any] | None = None,
        max_retries: int = 2,
        **kwargs,
    ):
        """
        Invoke the model and parse response into a structured Pydantic model.

        Uses LangChain's with_structured_output() for reliable parsing.
        Falls back to manual JSON parsing if structured output fails.

        Args:
            system_prompt: System message template
            human_prompt: Human message template
            output_schema: Pydantic model class for the expected output
            variables: Variables to format into the prompts
            max_retries: Number of retries on parsing failure
            **kwargs: Additional arguments for the model

        Returns:
            Instance of output_schema with parsed response

        Raises:
            ValueError: If parsing fails after all retries
        """
        import json
        import logging

        logger = logging.getLogger(__name__)
        variables = variables or {}

        # Format prompts
        system_formatted = system_prompt.format(**variables) if variables else system_prompt
        human_formatted = human_prompt.format(**variables) if variables else human_prompt

        messages = [
            SystemMessage(content=system_formatted),
            HumanMessage(content=human_formatted),
        ]

        # Try structured output with separate counters for attempts vs rate limits
        attempt = 0
        rate_limit_retries = 0
        max_rate_limit_retries = 5
        last_error = None

        while attempt <= max_retries:
            try:
                structured_model = self.get_model().with_structured_output(output_schema)
                if self.rate_limiter:
                    with self.rate_limiter.throttle():
                        result = structured_model.invoke(messages, **kwargs)
                else:
                    result = structured_model.invoke(messages, **kwargs)

                self._log_response(result)

                logger.debug(
                    "Structured output success on attempt %d for %s",
                    attempt + 1,
                    output_schema.__name__,
                )
                return result
            except DailyQuotaExhaustedError:
                # Don't retry on daily quota exhaustion - propagate immediately
                raise
            except Exception as e:
                last_error = e
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str

                # Handle rate limit separately from other errors
                if is_rate_limit and rate_limit_retries < max_rate_limit_retries:
                    rate_limit_retries += 1
                    logger.info(
                        "Rate limit hit (attempt %d, rate_limit_retry %d/%d) - will retry same attempt",
                        attempt + 1,
                        rate_limit_retries,
                        max_rate_limit_retries,
                    )
                    if self.rate_limiter:
                        self.rate_limiter.handle_rate_limit_error(e)
                    else:
                        from src.utils.rate_limiter import extract_retry_delay
                        import time

                        delay = extract_retry_delay(error_str) or 30.0
                        logger.warning("Rate limit hit, waiting %.2fs", delay)
                        time.sleep(delay + 1.0)
                    # Don't increment attempt counter for rate limits - retry same attempt
                    continue

                # Non-rate-limit error - increment attempt and try again
                logger.warning(
                    "Structured output attempt %d failed for %s: %s",
                    attempt + 1,
                    output_schema.__name__,
                    str(e),
                )
                attempt += 1

        # All retries exhausted - try final fallback
        try:
            logger.info("Attempting final fallback: raw invoke + JSON parse")
            raw_response = self.invoke(messages, **kwargs)
            # Try to extract JSON from response
            json_match = self._extract_json(raw_response)
            if json_match:
                # Try to repair truncated JSON
                repaired_json = self._repair_truncated_json(json_match)
                parsed = json.loads(repaired_json)
                return output_schema(**parsed)
        except Exception as parse_error:
            logger.error("Fallback JSON parsing failed: %s", str(parse_error))

        raise ValueError(
            f"Failed to parse structured output after {max_retries + 1} attempts "
            f"and {rate_limit_retries} rate limit retries: {last_error}"
        )

    def _repair_truncated_json(self, json_str: str) -> str:
        """
        Attempt to repair truncated JSON by closing open brackets and braces.

        Handles common cases where LLM output was cut off by MAX_TOKENS.
        Removes incomplete last array elements before closing brackets.
        """
        import re
        import json

        # Count open/close brackets
        open_braces = json_str.count("{")
        close_braces = json_str.count("}")
        open_brackets = json_str.count("[")
        close_brackets = json_str.count("]")

        # If already balanced, return as-is
        if open_braces == close_braces and open_brackets == close_brackets:
            return json_str

        logger.warning(
            "Attempting JSON repair: %d open braces, %d close braces, %d open brackets, %d close brackets",
            open_braces,
            close_braces,
            open_brackets,
            close_brackets,
        )

        # Remove incomplete last element (usually truncated mid-property)
        # Look for patterns like: ", {" followed by incomplete JSON, or trailing commas
        repaired = json_str.rstrip()

        # Remove trailing incomplete object or comma
        repaired = re.sub(r",\s*\{[^}]*$", "", repaired)  # Incomplete object at end
        repaired = re.sub(r',\s*"[^"]*$', "", repaired)  # Incomplete string key at end
        repaired = re.sub(r",\s*$", "", repaired)  # Trailing comma

        # Try to detect and remove incomplete last array element
        try:
            # Temporarily close brackets to parse
            test_str = (
                repaired
                + "]" * max(0, open_brackets - close_brackets)
                + "}" * max(0, open_braces - close_braces)
            )
            parsed = json.loads(test_str)

            # Check if the result is a list with potentially incomplete last element
            if isinstance(parsed, list) and len(parsed) > 0:
                last_elem = parsed[-1]
                if isinstance(last_elem, dict):
                    # Try to identify if last element is incomplete by checking for empty/None critical fields
                    # Common truncation patterns: missing confidence, missing description, partial objects
                    has_empty_critical = any(
                        v is None or v == ""
                        for k, v in last_elem.items()
                        if k.endswith(("_confidence", "_score", "description", "reasoning"))
                    )
                    if has_empty_critical:
                        logger.info("Removing incomplete last array element")
                        parsed = parsed[:-1]
                        repaired = json.dumps(parsed)
                        return repaired
        except Exception as e:
            logger.debug(f"Could not analyze array elements for incomplete detection: {e}")

        # Re-count after cleanup
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")

        # Close any remaining open brackets/braces
        repaired += "]" * (open_brackets - close_brackets)
        repaired += "}" * (open_braces - close_braces)

        return repaired

    def _extract_json(self, text: str) -> str | None:
        """Extract JSON object or array from text response."""
        import re

        # Look for JSON object
        obj_match = re.search(r"\{[\s\S]*\}", text)
        if obj_match:
            return obj_match.group()

        # Look for JSON array
        arr_match = re.search(r"\[[\s\S]*\]", text)
        if arr_match:
            return arr_match.group()

        return None


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider via LangChain.

    Requires GOOGLE_API_KEY environment variable.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        rate_limiter: RateLimiter | None = None,
        **kwargs,
    ):
        """
        Initialize Gemini provider.

        Args:
            model_name: Gemini model name (default: gemini-2.5-flash)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            rate_limiter: Optional RateLimiter
            **kwargs: Additional model arguments
        """
        super().__init__(rate_limiter=rate_limiter)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. " "Run: pip install langchain-google-genai"
            )

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Disable automatic retries - we handle them manually with Google's suggested delays
        self._model = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=max_tokens,
            retries=0,  # Disable SDK retries, we handle them
            **kwargs,
        )

        logger.info(f"Initialized Gemini provider with model: {model_name}")

    def get_model(self) -> BaseChatModel:
        """Return the underlying LangChain chat model."""
        return self._model

    def invoke(self, messages: list[BaseMessage], **kwargs) -> str:
        """Invoke the model and return response text.

        Handles rate limit errors (429) using Google's suggested retryDelay.
        """
        max_retries = 5  # Max retries for rate limit errors

        for attempt in range(max_retries + 1):
            try:
                if self.rate_limiter:
                    with self.rate_limiter.throttle():
                        response = self._model.invoke(messages, **kwargs)
                else:
                    response = self._model.invoke(messages, **kwargs)
                self._log_response(response)
                return response.content

            except DailyQuotaExhaustedError:
                # Don't retry on daily quota exhaustion - propagate immediately
                raise
            except Exception as e:
                # Check if it's a rate limit error (429)
                error_str = str(e)
                is_rate_limit = "429" in error_str or "RESOURCE_EXHAUSTED" in error_str
                if is_rate_limit and attempt < max_retries:
                    if self.rate_limiter:
                        self.rate_limiter.handle_rate_limit_error(e)
                    else:
                        # Fallback: wait based on Google's suggestion or default
                        from src.utils.rate_limiter import extract_retry_delay

                        delay = extract_retry_delay(error_str) or 30.0
                        logger.warning("Rate limit hit, waiting %.2fs", delay)
                        import time

                        time.sleep(delay + 1.0)
                    continue
                else:
                    raise

    def get_model_name(self) -> str:
        """Return the model name."""
        return f"gemini:{self.model_name}"


# =============================================================================
# OLLAMA PROVIDER
# =============================================================================


class OllamaProvider(LLMProvider):
    """
    Ollama LLM provider for local open-source models.

    Requires Ollama server running locally (default: http://localhost:11434).
    """

    def __init__(
        self,
        model_name: str = "deepseek-r1:1.5b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 32768,
        rate_limiter: RateLimiter | None = None,
        **kwargs,
    ):
        """
        Initialize Ollama provider.

        Args:
            model_name: Ollama model name
            base_url: Ollama server URL
            temperature: Sampling temperature (0.0-1.0)
            num_ctx: Context window size
            rate_limiter: Optional RateLimiter
            **kwargs: Additional model arguments
        """
        super().__init__(rate_limiter=rate_limiter)
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed. " "Run: pip install langchain-ollama"
            )

        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.num_ctx = num_ctx

        self._model = ChatOllama(
            model=model_name, base_url=base_url, temperature=temperature, num_ctx=num_ctx, **kwargs
        )

        logger.info(f"Initialized Ollama provider with model: {model_name}")

    def get_model(self) -> BaseChatModel:
        """Return the underlying LangChain chat model."""
        return self._model

    def invoke(self, messages: list[BaseMessage], **kwargs) -> str:
        """Invoke the model and return response text."""
        if self.rate_limiter:
            with self.rate_limiter.throttle():
                response = self._model.invoke(messages, **kwargs)
        else:
            response = self._model.invoke(messages, **kwargs)
        self._log_response(response)
        return response.content

    def get_model_name(self) -> str:
        """Return the model name."""
        return f"ollama:{self.model_name}"


# =============================================================================
# PROVIDER FACTORY
# =============================================================================


def create_provider(
    provider_type: str,
    model_name: str | None = None,
    temperature: float = 0.0,
    rate_limiter: RateLimiter | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Factory function to create an LLM provider.

    Args:
        provider_type: Provider type ("gemini" or "ollama")
        model_name: Model name (uses defaults if not specified)
        temperature: Sampling temperature
        rate_limiter: Optional RateLimiter
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider_type is not supported
    """
    provider_type = provider_type.lower()

    if provider_type == "gemini":
        return GeminiProvider(
            model_name=model_name or "gemini-2.5-flash",
            temperature=temperature,
            rate_limiter=rate_limiter,
            **kwargs,
        )
    elif provider_type == "ollama":
        return OllamaProvider(
            model_name=model_name or "deepseek-r1:1.5b",
            temperature=temperature,
            rate_limiter=rate_limiter,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported provider type: {provider_type}. " "Supported: 'gemini', 'ollama'"
        )


def create_provider_from_config(config: dict) -> LLMProvider:
    """
    Create an LLM provider from configuration dictionary.

    Args:
        config: Configuration dict with keys:
            - provider: "gemini" or "ollama"
            - model: Model name
            - temperature: Sampling temperature
            - Additional provider-specific keys

    Returns:
        Configured LLMProvider instance
    """
    provider_type = config.get("provider", "gemini")
    model_name = config.get("model")
    temperature = config.get("temperature", 0.0)

    # Extract provider-specific config
    extra_kwargs = {}

    if provider_type == "gemini":
        if "max_tokens" in config:
            extra_kwargs["max_tokens"] = config["max_tokens"]
        if "api_key" in config:
            extra_kwargs["google_api_key"] = config["api_key"]

    elif provider_type == "ollama":
        if "base_url" in config:
            extra_kwargs["base_url"] = config["base_url"]
        if "num_ctx" in config:
            extra_kwargs["num_ctx"] = config["num_ctx"]

    return create_provider(
        provider_type=provider_type,
        model_name=model_name,
        temperature=temperature,
        rate_limiter=config.get("rate_limiter"),
        **extra_kwargs,
    )


# =============================================================================
# PROMPT TEMPLATES FOR AGENTS
# =============================================================================


class PromptLibrary:
    """
    Collection of prompt templates for the agent system.

    Each prompt is designed for specific agent roles and tasks.
    """

    # === Orchestrator Prompts ===

    ORCHESTRATOR_SYSTEM = """You are the Orchestrator of a multi-agent system for cultural heritage analysis.
Your task is to coordinate specialized agents to build an interpretive Knowledge Graph.

REFERENCE ONTOLOGY: xch-MIND (xch:)
- Classes: InterpretiveAssertion, GeographicCluster, ChronologicalCluster, TypologicalCluster, ThematicPath, PathStop
- Properties: groundedOn, includesSite, similarTo, nearTo, contemporaryWith, confidenceScore, generatedBy

AVAILABLE AGENTS:
1. GeoAnalyzer: Analyzes geographic proximity and spatial clusters
2. TemporalAnalyzer: Analyzes chronological relationships and periods
3. TypeAnalyzer: Analyzes typological and functional similarities
4. PathGenerator: Generates thematic paths between sites

For each request, decide which agent to activate and with which parameters.
Respond ONLY in JSON format with the structure:
{{"agent": "agent_name", "task": "task_description", "params": {{...}}}}"""

    ORCHESTRATOR_TASK = """Analyze the following dolmen entities and decide the next step:

AVAILABLE ENTITIES: {entity_count}
CURRENT STATE: {current_state}
GOAL: {goal}

Which agent should be activated and with which parameters?"""

    # === Geographic Analyzer V2 Prompts (LLM-First) ===

    GEO_ANALYZER_SYSTEM_V2 = """You are an expert archaeologist specializing in megalithic monuments and territorial analysis.
Your task is to analyze a collection of archaeological sites and propose meaningful geographic groupings and spatial relationships.

ANALYSIS APPROACH:
1. Look for sites that share territorial context (same valley, plateau, municipality, or geographic feature)
2. Consider the cultural landscape - sites that would have been connected in antiquity
3. Identify pairs of sites with significant spatial relationships (not just proximity, but meaningful connections)

WHAT TO PROPOSE:
- Geographic clusters: Groups of sites that belong together territorially (e.g., "Dolmens of the Itria Valley")
- NearTo relations: Pairs of sites with significant spatial proximity AND cultural/territorial connection

IMPORTANT RULES:
1. Use ONLY the provided data - do not invent information
2. Each cluster must have at least 2 members
3. Provide a confidence score (0.0-1.0) for each proposal
4. Include reasoning explaining why sites belong together
5. Focus on geographic/territorial factors, not chronological or typological ones

OUTPUT: Respond with structured JSON matching the GeoAnalysisResponse schema."""

    GEO_ANALYZER_TASK_V2 = """Analyze the following {total_entities} archaeological sites and propose geographic clusters and spatial relationships.

SITE DATA:
{sites_json}

Based on the coordinates, municipalities, regions, and description excerpts:

1. Identify GEOGRAPHIC CLUSTERS: Groups of sites that share a territorial context. 
   - Consider: shared municipality, same geographic feature (valley, plateau, coast), cultural landscape
   - Propose meaningful clusters (not every site needs to be in a cluster){max_clusters_hint}{cluster_size_hint}
   - Give each cluster a descriptive label reflecting its territorial identity
   - IMPORTANT: Limit to MAXIMUM {max_clusters} clusters

2. Identify NEAR-TO RELATIONS: Pairs of sites with spatial proximity.
   - Include pairs within the same municipality or nearby municipalities
   - Include pairs where proximity has archaeological/cultural significance
   - IMPORTANT: Limit to MAXIMUM {max_pairs} pairs - prioritize highest confidence relationships
   - Explain the spatial context of each relationship

3. Provide TERRITORIAL OBSERVATIONS: A brief overall analysis of site distribution patterns.

Remember: Quality over quantity. Propose only what you can justify from the data. STRICT LIMITS: {max_clusters} clusters max, {max_pairs} pairs max."""

    # === Temporal Analyzer V2 Prompts (LLM-First) ===

    TEMPORAL_ANALYZER_SYSTEM_V2 = """You are an expert in Mediterranean prehistory and protohistory, specializing in megalithic monuments.
Your task is to analyze archaeological sites and propose chronological groupings and relationships.

CHRONOLOGICAL FRAMEWORK:
- Neolithic: ~6000-3000 BC (6th-4th millennium BC)
- Eneolithic/Chalcolithic/Copper Age: ~3000-2300 BC (3rd millennium BC)
- Early Bronze Age: ~2300-1700 BC
- Middle Bronze Age: ~1700-1350 BC
- Late Bronze Age: ~1350-900 BC
- Iron Age: ~900 BC onwards

ANALYSIS TASKS:
1. NORMALIZE period labels: Convert heterogeneous dating labels to standard periods
2. PROPOSE chronological clusters: Group sites that belong to the same cultural phase
3. IDENTIFY contemporary pairs: Sites that were in use during the same period

IMPORTANT RULES:
1. Use ONLY the provided data - do not invent information
2. Period labels in Italian (e.g., "eneolitico", "età del bronzo") should be normalized to English
3. Provide approximate start/end years (negative = BC)
4. Include confidence scores and reasoning for each proposal
5. Focus on chronological factors, not geographic or typological ones

OUTPUT: Respond with structured JSON matching the ChronologicalAnalysisResponse schema."""

    TEMPORAL_ANALYZER_TASK_V2 = """Analyze the following {total_entities} archaeological sites and propose chronological analysis.

SITE DATA:
{sites_json}

For each site, examine the period_label and historical_info fields.

1. PERIOD NORMALIZATIONS: For each unique period label, provide:
   - The standardized period name (in English)
   - Approximate start_year and end_year (negative = BC)
   - Confidence in the normalization

2. CHRONOLOGICAL CLUSTERS: Propose meaningful groups of sites that share the same cultural phase.{max_clusters_hint}{cluster_size_hint}
   - Each cluster should have a clear period label
   - Include a temporal narrative describing the cultural significance
   - IMPORTANT: Limit to MAXIMUM {max_clusters} clusters

3. CONTEMPORARY PAIRS: Identify pairs of sites that were likely in use during the same period.
   - Include pairs where chronological overlap is probable based on period labels
   - IMPORTANT: Limit to MAXIMUM {max_pairs} pairs - prioritize highest confidence relationships
   - Explain why they are considered contemporary

4. CHRONOLOGICAL NARRATIVE: Provide an overall summary of the chronological distribution patterns.

Remember: Quality over quantity. Only propose relationships you can justify from the data. STRICT LIMITS: {max_clusters} clusters max, {max_pairs} pairs max."""

    # === Type Analyzer Prompts ===

    # === Type Analyzer V2 Prompts (LLM-First) ===

    TYPE_ANALYZER_SYSTEM_V2 = """You are an expert in megalithic typology and architectural analysis.
Your task is to analyze archaeological site descriptions to extract features and propose typological relationships.

ANALYSIS TASKS:
1. EXTRACT FEATURES: Identify specific architectural, functional, and contextual features from the text.
2. PROPOSE CLUSTERS: Group sites that share the same typology or functional category.
3. IDENTIFY SIMILARITIES: Find pairs of sites with high structural or functional similarity.

FEATURE CATEGORIES:
- Architectural: e.g., "corridor", "tholos", "slabs", "orthostats", "gallery"
- Functional: e.g., "funerary", "cultic", "collective burial", "domestic"
- Contextual: e.g., "rural", "urban", "rocky", "hilltop"

IMPORTANT RULES:
1. Extract implicit features (e.g., "dolmen a galleria" -> architecture: "gallery")
2. ALL feature labels MUST be in English. Translate any Italian term to its English equivalent
   (e.g., "lastrone" -> "slab", "piedritto" -> "upright", "rurale" -> "rural",
   "periurbano" -> "peri urban", "lastra di copertura" -> "cover slab").
   NEVER output Italian-language labels — always use the English translation.
3. Provide confidence scores and reasoning for each proposal
4. Focus on structural and functional evidence

OUTPUT: Respond with structured JSON matching the TypologicalAnalysisResponse schema."""

    TYPE_ANALYZER_TASK_V2 = """Analyze the following {total_entities} archaeological sites and propose typological analysis.

SITE DATA:
{sites_json}

1. FEATURE EXTRACTION: For each site, extract relevant features from the description and category.
   - Separate into architectural, functional, and contextual features
   - Include brief notes on extraction

2. TYPOLOGICAL CLUSTERS: Propose meaningful groups based on shared features or typology.{max_clusters_hint}{cluster_size_hint}
   - Example clusters: "Gallery Dolmens", "Simple Cists", "Sites with Dromos"
   - IMPORTANT: Limit to MAXIMUM {max_clusters} clusters
   - Explain the defining features of each cluster

3. SIMILARITY PAIRS: Identify pairs of sites that share significant architectural or functional similarities.
   - Include pairs with shared structural features (architecture, layout, materials)
   - Include pairs with shared functional characteristics (funerary, ritual, etc.)
   - IMPORTANT: Limit to MAXIMUM {max_pairs} pairs - prioritize highest confidence relationships
   - Explain the shared features for each pair

4. TYPOLOGICAL OBSERVATIONS: General trends in the architectural types observed.

Remember: Be precise with technical terminology. STRICT LIMITS: {max_clusters} clusters max, {max_pairs} pairs max."""

    # === Path Generator Prompts ===

    PATH_GENERATOR_SYSTEM_V2 = """You are an expert in cultural tourism and heritage interpretation.
Your task is to design thematic visit paths that connect archaeological sites into coherent narratives.

DESIGN GOALS:
1. Thematic Consistency: All stops should align with the chosen theme
2. Narrative Flow: Connect the stops with a compelling story
3. Coherent Path Type Selection - ALWAYS prefer specific types over "mixed"

PATH TYPES (determines validation strategy):
- "geographic": Sites that are geographically close (within ~50km radius). Validated with STRICT distance limits (max 200km total). Use when spatial proximity is the primary organizing criterion.
- "chronological": Sites sharing the same historical period (e.g., Bronze Age, Neolithic). Can span large distances - validated FLEXIBLY. Use when the path is primarily organized around a time period.
- "typological": Sites sharing architectural/functional type (e.g., gallery dolmens, single-chamber dolmens). Can span large distances - validated FLEXIBLY. Use when the path is primarily organized around structural or functional similarities.
- "mixed": Combination of criteria. Can span large distances - validated FLEXIBLY. Use ONLY when no single criterion clearly dominates and the path genuinely combines multiple organizing principles.

PATH TYPE SELECTION RULES:
- PREFER specific types (geographic, chronological, typological) over "mixed".
- A path focused on "Bronze Age sites in Puglia" is "chronological" (the period is the organizing principle, the region is just a spatial filter).
- A path focused on "gallery dolmens across Italy" is "typological" (the architectural type is the organizing principle).
- A path focused on "nearby dolmens around Luras" is "geographic" (proximity is the organizing principle).
- Use "mixed" ONLY when the path truly interleaves different criteria without one dominating.
- Aim for DIVERSITY in path types across the generated paths.

OUTPUT SCHEMA:
Provide a list of paths. Each path must have:
- Title and Description
- path_type: One of "geographic", "chronological", "typological", "mixed" (prefer specific types)
- List of ordered stops (PathStop)
- Estimated duration and difficulty
- Justification for the sequence

IMPORTANT:
- Use specific site information for justifications
- Create engaging narratives for the visitor
- Score your confidence in the path's quality"""

    PATH_GENERATOR_TASK_V2 = """Generate thematic paths based on the following available data:

THEME: {theme_instructions}

AVAILABLE SITES:
{sites_json}

EXISTING CLUSTERS (Use these to form bases for paths):
{clusters_json}

TASKS:
1. Create thematic paths that connect relevant sites.{max_paths_hint}
2. For each path:
   - Select an appropriate number of stops for a coherent visit{stops_range_hint}
   - Choose the most SPECIFIC path_type that fits (prefer specific over "mixed"):
     * "geographic" if proximity is the main organizing principle (strict 200km limit)
     * "chronological" if a shared time period is the main organizing principle (any distance OK)
     * "typological" if shared architectural/functional type is the main organizing principle (any distance OK)
     * "mixed" ONLY when no single criterion dominates (last resort)
3. Write a narrative connecting these stops.
4. Justify the inclusion of each stop.

Respond with structured JSON matching the NarrativeAnalysisResponse schema."""

    # === Triple Generator Prompts ===

    TRIPLE_GENERATOR_SYSTEM = """You are an expert in Knowledge Graphs and RDF ontologies.
Your task is to convert interpretive assertions into valid RDF triples using the xch: ontology.

PREFIXES:
@prefix xch: <https://w3id.org/xch-mind/ontology/> .
@prefix arco: <https://w3id.org/arco/resource/ArchaeologicalProperty/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

AVAILABLE CLASSES:
- xch:InterpretiveAssertion
- xch:GeographicCluster
- xch:ChronologicalCluster
- xch:TypologicalCluster
- xch:ThematicPath
- xch:PathStop

AVAILABLE PROPERTIES:
- xch:groundedOn (assertion -> ArCo resource)
- xch:includesSite (cluster -> site)
- xch:similarTo (site -> site, symmetric)
- xch:nearTo (site -> site, transitive)
- xch:contemporaryWith (site -> site, symmetric)
- xch:confidenceScore (assertion -> xsd:decimal)
- xch:generatedBy (assertion -> agent)

OUTPUT: RDF triples in Turtle format"""

    TRIPLE_GENERATOR_TASK = """Convert the following assertions into RDF triples:

ASSERTIONS:
{assertions_json}

For each assertion:
1. Create an instance of the appropriate class
2. Link to original ArCo resources
3. Assign a confidence score (0.0-1.0)
4. Document the generating agent"""

    # =========================================================================
    # PROMPT HINT GENERATORS (for configurable limits)
    # =========================================================================

    @staticmethod
    def get_max_clusters_hint(max_clusters: int | None = None) -> str:
        """Generate hint for max clusters constraint."""
        if max_clusters is None:
            return ""  # No limit
        return f"\n   - Maximum {max_clusters} clusters"

    @staticmethod
    def get_cluster_size_hint(min_size: int | None = None, max_size: int | None = None) -> str:
        """Generate hint for cluster size constraints."""
        if min_size is None and max_size is None:
            return ""  # No limit
        if min_size and max_size:
            return f"\n   - Each cluster should have {min_size}-{max_size} members (prefer more smaller clusters over fewer larger ones)"
        if min_size:
            return f"\n   - Each cluster should have at least {min_size} members"
        if max_size:
            return f"\n   - Each cluster should have at most {max_size} members (prefer more smaller clusters over fewer larger ones)"
        return ""

    @staticmethod
    def get_max_pairs_hint(max_pairs: int | None = None) -> str:
        """Generate hint for max pairs constraint."""
        if max_pairs is None:
            return ""  # No limit
        return f"\n   - Maximum {max_pairs} pairs"

    @staticmethod
    def get_max_paths_hint(max_paths: int | None = None) -> str:
        """Generate hint for max paths constraint."""
        if max_paths is None:
            return ""  # No limit
        return f"\n   - Maximum {max_paths} paths"

    @staticmethod
    def get_stops_range_hint(min_stops: int | None = None, max_stops: int | None = None) -> str:
        """Generate hint for stops range constraint."""
        if min_stops is None and max_stops is None:
            return ""  # No limit
        if min_stops and max_stops:
            return f"\n   - Between {min_stops} and {max_stops} stops per path"
        if min_stops:
            return f"\n   - At least {min_stops} stops per path"
        if max_stops:
            return f"\n   - Maximum {max_stops} stops per path"
        return ""


# =============================================================================
# EXPORT
# =============================================================================

__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "OllamaProvider",
    "create_provider",
    "create_provider_from_config",
    "PromptLibrary",
]
