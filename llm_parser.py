"""
LLM 需求解析器 — 替代简单的关键词规则匹配

支持任意 OpenAI 兼容接口（OpenAI、DeepSeek、Qwen、本地 vLLM/Ollama 等）
失败时自动回退到规则解析器 (RequirementParser)
"""
import json
import os
from urllib import request, error
from dataclasses import dataclass, field

from model import RequirementVector


# 从 .env 文件加载环境变量（同时兼容 os.getenv）
_DOTENV_CACHE = {}


def _load_dotenv():
    """读取项目根目录 .env 文件到缓存"""
    global _DOTENV_CACHE
    if _DOTENV_CACHE:
        return _DOTENV_CACHE
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.isfile(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                _DOTENV_CACHE[key.strip()] = val.strip()
    return _DOTENV_CACHE


def _env(key: str, default: str = "") -> str:
    """先查系统环境变量，再查 .env 文件"""
    val = os.getenv(key)
    if val is not None:
        return val
    return _load_dotenv().get(key, default)


# ============================================================
# 1. LLM 解析器
# ============================================================

LLM_SYSTEM_PROMPT = """你是一位军事作战需求分析专家。请将用户的自然语言需求解析为结构化的权重向量和约束条件。

输出要求：
1. w_kill（杀伤效能权重）：用户对摧毁/打击目标的重视程度，0.0~1.0
2. w_cost（成本控制权重）：用户对弹药/资源消耗的敏感程度，0.0~1.0
3. w_time（时间效率权重）：用户对快速响应、时敏目标的重视程度，0.0~1.0
4. w_info（信息优势权重）：用户对情报/态势感知的重视程度，0.0~1.0
5. w_network（网络质量权重）：用户对通信/链路质量的重视程度，0.0~1.0
6. target_priority（目标优先级）："high_value"（高价值优先）、"time_sensitive"（时敏优先）或 "all"（无特殊偏好）
7. constraints（约束条件）：提取具体的数值约束

约束条件映射规则：
- 成本约束："500万" → max_cost=500，"1000万" → max_cost=1000，默认1000
- 时间约束："3分钟" → time_window=180，"5分钟" → time_window=300，默认300
- 杀伤概率约束："杀伤概率0.8" → min_kill_prob=0.8，"杀伤概率0.9" → min_kill_prob=0.9，默认0.5

请只返回严格合法的 JSON，不要包含任何解释文字。格式示例：
{
    "w_kill": 0.85,
    "w_cost": 0.30,
    "w_time": 0.60,
    "w_info": 0.20,
    "w_network": 0.15,
    "target_priority": "high_value",
    "constraints": {"max_cost": 500, "time_window": 180}
}"""


@dataclass
class LLMConfig:
    """LLM API 配置"""
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = "sk-ed9a4ce0d9394a6f8880a712a380df7a"
    model: str = "deepseek-chat"
    timeout: int = 15


class LLMRequirementParser:
    """基于 LLM 的需求解析器

    用法:
        config = LLMConfig(base_url="...", api_key="...", model="...")
        vec = LLMRequirementParser.parse("最大化杀伤高价值目标，控制成本", config)
    """

    @classmethod
    def parse(cls, requirement: str, config=None) -> RequirementVector:
        """调用 LLM 解析需求，失败时回退到规则解析器

        config 支持 LLMConfig 对象或 dict（如 {"base_url": ..., "api_key": ...}）
        """
        config = cls._normalize_config(config)
        if not config.api_key:
            raise RuntimeError("未配置 LLM API Key，请在侧边栏填写或设置环境变量 LLM_API_KEY")

        try:
            return cls._call_llm(requirement, config)
        except Exception as e:
            # LLM 失败时抛出异常，让调用方决定是否回退
            raise RuntimeError(f"LLM 解析失败: {e}")

    @classmethod
    def parse_with_fallback(cls, requirement: str, config=None) -> RequirementVector:
        """调用 LLM 解析需求，失败时自动回退到规则解析器

        config 支持 LLMConfig 对象或 dict（如 {"base_url": ..., "api_key": ...}）
        """
        from model import RequirementParser
        try:
            return cls.parse(requirement, config)
        except Exception:
            return RequirementParser.parse(requirement)

    @classmethod
    def _normalize_config(cls, config) -> LLMConfig:
        """统一把 dict 或 None 转成 LLMConfig 对象"""
        if config is None:
            return cls._config_from_env()
        if isinstance(config, LLMConfig):
            return config
        if isinstance(config, dict):
            return LLMConfig(
                base_url=config.get("base_url", "https://api.deepseek.com/v1"),
                api_key=config.get("api_key", ""),
                model=config.get("model", "deepseek-chat"),
                timeout=int(config.get("timeout", 15)),
            )
        # 其他未知类型回退到环境变量
        return cls._config_from_env()

    @classmethod
    def _config_from_env(cls) -> LLMConfig:
        """从环境变量 / .env 文件读取配置"""
        return LLMConfig(
            base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=_env("LLM_API_KEY", ""),
            model=_env("LLM_MODEL", "deepseek-chat"),
        )

    @classmethod
    def _call_llm(cls, requirement: str, config: LLMConfig) -> RequirementVector:
        """通过 HTTP 调用 OpenAI 兼容接口"""
        url = config.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
        payload = {
            "model": config.model,
            "messages": [
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": f"用户作战需求：{requirement}"},
            ],
            "temperature": 0.3,
            "max_tokens": 512,
        }

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=data, headers=headers, method="POST")

        try:
            with request.urlopen(req, timeout=config.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code}: {body}")

        # 提取 LLM 回复内容
        content = result["choices"][0]["message"]["content"]
        # 清理可能的 markdown 代码块
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        parsed = json.loads(content)
        return cls._json_to_vector(parsed)

    @classmethod
    def _json_to_vector(cls, data: dict) -> RequirementVector:
        """将 LLM 返回的 JSON 转换为 RequirementVector"""
        constraints = {}
        raw_constraints = data.get("constraints", {})
        if isinstance(raw_constraints.get("max_cost"), (int, float)):
            if raw_constraints["max_cost"] == 500:
                constraints["max_cost_500"] = True
            elif raw_constraints["max_cost"] == 1000:
                constraints["max_cost_1000"] = True
        if isinstance(raw_constraints.get("time_window"), (int, float)):
            if raw_constraints["time_window"] == 180:
                constraints["time_window_180"] = True
            elif raw_constraints["time_window"] == 300:
                constraints["time_window_300"] = True
        if isinstance(raw_constraints.get("min_kill_prob"), (int, float)):
            if raw_constraints["min_kill_prob"] == 0.8:
                constraints["min_kill_08"] = True
            elif raw_constraints["min_kill_prob"] == 0.9:
                constraints["min_kill_09"] = True

        return RequirementVector(
            w_kill=float(data.get("w_kill", 0.5)),
            w_cost=float(data.get("w_cost", 0.5)),
            w_time=float(data.get("w_time", 0.5)),
            w_info=float(data.get("w_info", 0.5)),
            w_network=float(data.get("w_network", 0.5)),
            target_priority=data.get("target_priority", "all"),
            constraints=constraints,
        )
