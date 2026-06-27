# config.py
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    YamlConfigSettingsSource,
)

# 1. 定义嵌套的子配置模块 (继承 BaseModel)
class ModelConfig(BaseModel):
    path: str = Field(default="./DS-R1-Distill-Qwen-1.5B", description="模型路径")
    eos_token_id: int = Field(description="模型的EOS Token ID")

class EngineConfig(BaseModel):
    block_size: int = Field(default=512, description="KV_cache分页的块大小")
    max_num_running_seqs: int = Field(default=64, description="引擎中批量运行的最大序列数")
    max_num_kvcache_blocks: int = Field(default=16, description="引擎中KV_cache的最大块数，根据GPU显存自动计算出的KV_cache块数，用户无需配置")
    max_num_running_batched_tokens: int = Field(default=8192, description="引擎中批量运行的最大token数，这个参数主要会影响运行时的激活值显存占用大小")
    max_model_len: int = Field(default=2048, description="单用户的最大输入长度，但我看nano-vllm中没用用这个参数限制输入prompt的长度，只用在model_runner的warmup中，所以暂时不考虑太多")
    gpu_memory_utilization: float = Field(default=0.9, description="GPU显存利用率上限，范围0-1，默认为0.9")

class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

# 2. 定义全局的主配置类 (继承 BaseSettings)
class AppSettings(BaseSettings):
    # 将上面的子模块作为字段引入
    model: ModelConfig
    engine: EngineConfig
    server: ServerConfig

    # 3. 核心：重写配置源，告诉 Pydantic 从 YAML 文件读取
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # 返回的数据源元组决定了优先级。排在前面的优先级更高。
        # 这里我们将 YamlConfigSettingsSource 放在首位
        return (
            YamlConfigSettingsSource(settings_cls, yaml_file='./config.yaml'),
            env_settings, # 保留环境变量覆盖的能力
        )

# 4. 实例化全局单例
# 只要在这个模块被 import，就会自动去读取 config.yaml 并完成校验
Config = AppSettings()