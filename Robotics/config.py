from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    base_url: str = "http://127.0.0.1:8001/v1"
    api_key: str = "not-needed"
    model_name: str = "local-model"
    temperature: float = 0.1
    max_tokens: int = 256
    timeout: float = 30.0


@dataclass
class JevConfig:
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        base_url="http://127.0.0.1:8001/v1",
        model_name="Jev-Style-0.8B-Decision-v3-GGUF",
        temperature=0.0,
        max_tokens=64,
    ))
    confidence_threshold: float = 0.85
    bins: dict = field(default_factory=lambda: {
        "A": "Heavy",
        "B": "Fragile",
        "C": "Standard",
        "D": "Recycle",
    })


@dataclass
class ReasonerConfig:
    llm: LLMConfig = field(default_factory=lambda: LLMConfig(
        base_url="http://127.0.0.1:8002/v1",
        model_name="mistral-7b-instruct-v0.2-Q5_K_M.gguf",
        temperature=0.3,
        max_tokens=512,
    ))


@dataclass
class MuJoCoConfig:
    scene_xml: str = "scenes/sorting_scene.xml"
    time_step: float = 0.002
    pick_height_offset: float = 0.08
    move_speed: float = 0.6
    grip_force: float = 50.0


@dataclass
class SystemConfig:
    jev: JevConfig = field(default_factory=JevConfig)
    reasoner: ReasonerConfig = field(default_factory=ReasonerConfig)
    mujoco: MuJoCoConfig = field(default_factory=MuJoCoConfig)