"""
LLM-based Motion Planner

Takes a multimodal context dict and produces a sequence of animated motion
primitives with animation-principle parameters.

Supports two pipelines:
  - **short** (single LLM call → sequence + principle ratings)
  - **long**  (three LLM calls: plan → descriptions → parameters)

The context dict is inherently multimodal:
  {"text": "...", "image": "path/to/img.png", "audio": "path/to/clip.wav"}
Any combination of keys is valid.
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Dict, List, Literal, Optional, Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError, conlist

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Backend-specific imports (deferred so the unused backend need not be installed)
try:
    from openai import APIConnectionError, APIError, OpenAI, RateLimitError
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]

try:
    import ollama as _ollama_mod
    from ollama import Client as OllamaClient
    from ollama import ResponseError as OllamaResponseError
except ImportError:
    OllamaClient = None  # type: ignore[assignment,misc]
    _ollama_mod = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════

class PrincipleScaleDefinition(BaseModel):
    scale_description: str
    scale_range: List[int]


class Animation(BaseModel):
    motion_primitive: str
    Anticipation: int
    Arcs: int
    Exaggeration: int
    Follow_Through: int
    Slow_In_Slow_Out: int
    Timing: int


class PlanWithContextInput(BaseModel):
    context: str
    robot_capabilities: str
    available_motion_primitives: Dict[str, str]
    principles: Dict[str, PrincipleScaleDefinition]


class PlanWithContextOutput(BaseModel):
    animated_sequence: List[Animation]


# Long-pipeline schemas
class ContextToSequenceInput(BaseModel):
    context: str
    robot_capabilities: str
    available_motion_primitives: Dict[str, str]


class ContextToSequenceOutput(BaseModel):
    motion_primitive_sequence: List[str]


class AnimationDescription(BaseModel):
    motion: str
    description: List[str]


class ContextAndSequenceToAnimationDescriptionInput(BaseModel):
    motion_primitive_sequence: List[str]
    context: str


class ContextAndSequenceToAnimationDescriptionOutput(BaseModel):
    animation_descriptions: List[AnimationDescription]


class AnimationDescriptionToAnimationPrincipleDescriptionInput(BaseModel):
    motion_primitive: str
    animation_description: List[str]
    principles: Dict[str, PrincipleScaleDefinition]


class AnimationDescriptionToAnimationPrincipleDescriptionOutput(BaseModel):
    Anticipation: int
    Arcs: int
    Exaggeration: int
    Follow_Through: int
    Slow_In_Slow_Out: int
    Timing: int


# ═══════════════════════════════════════════════════════════════════════════════
# Planner
# ═══════════════════════════════════════════════════════════════════════════════

class Planner:
    """LLM-backed planner that turns multimodal context into animated motion sequences."""

    def __init__(
        self,
        robot,
        prompt_data_path: str = "prompts.yaml",
        llm_backend: Literal["openai", "ollama"] = "openai",
        openai_model: str = "gpt-4.1",
        ollama_model: str = "llama3.1",
        ollama_host: str = "http://localhost:11434",
    ) -> None:
        self.robot = robot
        self.prompt_data = self._load_yaml(prompt_data_path)
        self.llm_backend = llm_backend

        if llm_backend == "openai":
            if OpenAI is None:
                raise ImportError("openai package is not installed. Run: pip install openai")
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError(
                    "OPENAI_API_KEY not set. Export it or place it in a .env file."
                )
            self.openai_client = OpenAI()
            self.openai_model: str = openai_model
            logger.info("Planner initialised  backend=openai  model=%s", self.openai_model)

        elif llm_backend == "ollama":
            if OllamaClient is None:
                raise ImportError("ollama package is not installed. Run: pip install ollama")
            self.ollama_client = OllamaClient(host=ollama_host)
            self.ollama_model: str = ollama_model
            logger.info(
                "Planner initialised  backend=ollama  model=%s  host=%s",
                self.ollama_model, ollama_host,
            )
        else:
            raise ValueError(f"Unknown llm_backend: {llm_backend!r}. Use 'openai' or 'ollama'.")

    # ── public pipelines ─────────────────────────────────────────────────────

    def short_pipeline(self, context: dict) -> tuple[list[dict], list[dict]]:
        """Single-call pipeline: context → animated sequence."""
        return self._generate_animated_sequence(context)

    def long_pipeline(self, context: dict) -> tuple[list[dict], list[dict]]:
        """Three-call pipeline: plan → descriptions → parameters."""
        plan = self._generate_plan(context)
        descriptions = self._generate_animation_descriptions(plan, context)

        mapped_seq: list[dict] = []
        unmapped_seq: list[dict] = []
        for i, desc in enumerate(descriptions):
            logger.info("[planner] Parameterising %d/%d", i + 1, len(descriptions))
            mapped, unmapped = self._generate_animation_parameters(desc)
            mapped["motion_primitive"] = plan[i]
            unmapped["motion_primitive"] = plan[i]
            mapped_seq.append(mapped)
            unmapped_seq.append(unmapped)
        return mapped_seq, unmapped_seq

    # ── internal: short pipeline ─────────────────────────────────────────────

    def _generate_animated_sequence(self, context: dict) -> tuple[list[dict], list[dict]]:
        system_prompt = self.prompt_data["plan_with_context"]

        textual_context = self._build_textual_context(context)
        input_data = PlanWithContextInput(
            context=textual_context,
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib(),
            principles=self.prompt_data["principles"],
        )

        result = self._call_llm(
            system_prompt,
            input_data.model_dump(),
            PlanWithContextOutput,
            image_path=context.get("image"),
        )

        mapped_seq: list[dict] = []
        unmapped_seq: list[dict] = []
        for anim in result.animated_sequence:
            mapped = self._map_principles_to_parameters(anim.dict())
            mapped["motion_primitive"] = anim.motion_primitive
            mapped_seq.append(mapped)
            unmapped_seq.append(anim.dict())
        return mapped_seq, unmapped_seq

    # ── internal: long pipeline ──────────────────────────────────────────────

    def _generate_plan(self, context: dict) -> list[str]:
        textual_context = self._build_textual_context(context)
        input_data = ContextToSequenceInput(
            context=textual_context,
            robot_capabilities=self.robot.get_capabilities(),
            available_motion_primitives=self.robot.get_primitive_lib(),
        )
        result = self._call_llm(
            self.prompt_data["context_to_sequence"],
            input_data.model_dump(),
            ContextToSequenceOutput,
            image_path=context.get("image"),
        )
        return result.motion_primitive_sequence

    def _generate_animation_descriptions(self, plan: list[str], context: dict):
        textual_context = self._build_textual_context(context)
        input_data = ContextAndSequenceToAnimationDescriptionInput(
            motion_primitive_sequence=plan,
            context=textual_context,
        )
        result = self._call_llm(
            self.prompt_data["context_and_sequence_to_animation_description"],
            input_data.model_dump(),
            ContextAndSequenceToAnimationDescriptionOutput,
            image_path=context.get("image"),
        )
        return result.animation_descriptions

    def _generate_animation_parameters(self, description) -> tuple[dict, dict]:
        input_data = AnimationDescriptionToAnimationPrincipleDescriptionInput(
            motion_primitive=description.motion,
            animation_description=description.description,
            principles=self.prompt_data["principles"],
        )
        result = self._call_llm(
            self.prompt_data["animation_description_to_animation_principle_description"],
            input_data.model_dump(),
            AnimationDescriptionToAnimationPrincipleDescriptionOutput,
        )
        return self._map_principles_to_parameters(result.dict()), result.dict()

    # ── multimodal context assembly ──────────────────────────────────────────

    def _build_textual_context(self, context: dict) -> str:
        """Convert a multimodal context dict into a single textual description
        suitable for the LLM ``context`` field.

        - ``text``  → used verbatim
        - ``audio`` → transcribed via Whisper, then included as text
        - ``image`` → a note is appended; the actual image is passed separately
        """
        parts: list[str] = []
        modalities = [k for k in ("text", "image", "audio") if k in context]
        multimodal = len(modalities) > 1

        if multimodal:
            parts.append(
                "The context consists of multiple modalities that complement each other. "
                "Individual modalities are separated with a semicolon."
            )

        if "text" in context:
            prefix = "The text modality is as follows: " if multimodal else ""
            parts.append(f"{prefix}{context['text']}")

        if "audio" in context:
            transcription = self._transcribe_audio(context["audio"])
            if multimodal:
                parts.append(f"The transcribed audio modality is as follows: {transcription}")
            else:
                parts.append(f"The context is given from transcribed speech as follows: {transcription}")

        if "image" in context:
            if multimodal:
                parts.append("The image modality is given in the passed image for you to analyze.")
            else:
                parts.append("The context is given in the passed image for you to analyze.")

        return "; ".join(parts) if multimodal else " ".join(parts)

    def _transcribe_audio(self, audio_path: str) -> str:
        if self.llm_backend != "openai":
            raise RuntimeError(
                "Audio transcription is only supported with the 'openai' backend. "
                "Remove the 'audio' key from the context or switch to the OpenAI backend."
            )
        with open(audio_path, "rb") as f:
            result = self.openai_client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=f,
                prompt="Transcribe the contents. It can be speech or sounds.",
            )
        logger.info("[planner] Transcribed audio: %s", result.text)
        return result.text

    # ── parameter mapping ────────────────────────────────────────────────────

    def _map_principles_to_parameters(self, principle_desc: dict) -> dict:
        """Map abstract principle ratings to robot-specific parameter values."""
        mapped: dict = {}

        robot_ranges = self.robot.get_parameter_ranges()
        if "Follow_Through_Data" in robot_ranges:
            mapped["Follow_Through_Data"] = robot_ranges["Follow_Through_Data"]

        for key, value in principle_desc.items():
            if key == "motion_primitive":
                continue
            if key not in self.prompt_data["principles"]:
                raise KeyError(f"'{key}' not in prompt principles")

            # Boolean case
            if key == "Slow_In_Slow_Out":
                mapped[key] = bool(value)
                continue

            if key not in robot_ranges:
                raise KeyError(f"'{key}' not in robot parameter ranges")

            gen_min, gen_max = self.prompt_data["principles"][key]["scale_range"]
            spec_min = robot_ranges[key]["min"]
            spec_max = robot_ranges[key]["max"]

            if key in ("Timing", "Exaggeration", "Arcs"):
                neutral = 1.0
                if value == 0:
                    mapped[key] = neutral
                elif value < 0:
                    mapped[key] = neutral + (value / gen_min) * (spec_min - neutral)
                else:
                    mapped[key] = neutral + (value / gen_max) * (spec_max - neutral)
            elif key in ("Anticipation", "Follow_Through"):
                mapped[key] = spec_min + (value - gen_min) / (gen_max - gen_min) * (spec_max - spec_min)
            else:
                raise KeyError(f"No mapping rule for '{key}'")

        return mapped

    # ── LLM call helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_llm(
        self,
        system_prompt: str,
        input_data: dict,
        output_schema: Type[T],
        temperature: float = 0.0,
        image_path: Optional[str] = None,
    ) -> T:
        """Dispatch to the active backend."""
        if self.llm_backend == "openai":
            return self._call_llm_openai(
                system_prompt, input_data, output_schema,
                temperature=temperature, image_path=image_path,
            )
        else:
            return self._call_llm_ollama(
                system_prompt, input_data, output_schema,
                temperature=temperature, image_path=image_path,
            )

    # ── OpenAI backend ───────────────────────────────────────────────────────

    def _call_llm_openai(
        self,
        system_prompt: str,
        input_data: dict,
        output_schema: Type[T],
        temperature: float = 0.0,
        image_path: Optional[str] = None,
    ) -> T:
        user_content: list[dict] = [{"type": "input_text", "text": json.dumps(input_data)}]
        if image_path:
            b64 = self._encode_image(image_path)
            user_content.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        logger.info("[planner] Calling OpenAI  model=%s  image=%s", self.openai_model, bool(image_path))

        try:
            response = self.openai_client.responses.parse(
                model=self.openai_model,
                input=messages,
                temperature=temperature,
                text_format=output_schema,
            )
            return response.output_parsed
        except (APIError, APIConnectionError, RateLimitError) as e:
            logger.error("OpenAI API error: %s", e)
            raise

    # ── Ollama backend ───────────────────────────────────────────────────────

    def _call_llm_ollama(
        self,
        system_prompt: str,
        input_data: dict,
        output_schema: Type[T],
        temperature: float = 0.0,
        image_path: Optional[str] = None,
    ) -> T:
        # Build the user message content
        # Ollama expects 'content' as a plain string; images go in a separate 'images' list.
        user_text = json.dumps(input_data)

        # Hint the model to return JSON (improves reliability)
        user_text += "\n\nReturn your answer as JSON."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

        # Attach image if provided (Ollama vision models accept base64 images)
        if image_path:
            b64 = self._encode_image(image_path)
            messages[-1]["images"] = [b64]  # type: ignore[index]

        logger.info(
            "[planner] Calling Ollama  model=%s  image=%s",
            self.ollama_model, bool(image_path),
        )

        try:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=messages,
                format=output_schema.model_json_schema(),
                options={"temperature": temperature},
            )
        except Exception as e:
            logger.error("Ollama API error: %s", e)
            raise

        raw_content = response.message.content

        try:
            result = output_schema.model_validate_json(raw_content)
        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(
                "Failed to parse Ollama response into %s: %s\nRaw: %s",
                output_schema.__name__, e, raw_content,
            )
            raise ValueError(
                f"Ollama returned invalid structured output for {output_schema.__name__}: {e}"
            ) from e

        return result

    # ── misc ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_yaml(path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)
