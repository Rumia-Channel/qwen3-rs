#[cfg(test)]
#[path = "../tests/unit/config_loader_test.rs"]
mod config_loader_test;

use anyhow::{Context, Result};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, path::Path};

use crate::models::ArchitectureId;

/// Model type detection with embedded LoRA configuration
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Base,             // Standard base model
    LoRA(LoRAConfig), // LoRA fine-tuned model with full config
}

/// Enhanced model information that includes type and configs
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub config: ModelConfig,
}

/// Qwen3.5 extended configuration (only present for Qwen3.5 architecture).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// Chat EOS from tokenizer_config.json.
    pub chat_eos: u32,
    /// Linear attention key heads.
    pub n_linear_k_heads: u32,
    /// Linear attention value heads.
    pub n_linear_v_heads: u32,
    pub linear_k_head_dim: u32,
    pub linear_v_head_dim: u32,
    pub conv_kernel_size: u32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub full_attention_mask: u64,
}

/// Configuration structure matching the Python ModelArgs
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub vocab_size: u32,
    pub max_seq_len: u32,
    pub head_dim: u32,
    pub norm_eps: f32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub architecture: ArchitectureId,
    /// Qwen3.5 extended configuration (None for v1 architectures).
    pub qwen35: Option<Qwen35Config>,
}

/// LoRA configuration from adapter_config.json
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LoRAConfig {
    pub lora_alpha: f32,
    pub r: usize,
    pub target_modules: Vec<String>,
    pub base_model_name_or_path: Option<String>,
}

/// Auto-detect model type and load appropriate configuration
/// This is the main entry point that replaces load_hf_config
pub fn load_model_info(model_path: &str) -> Result<ModelInfo> {
    let model_path = Path::new(model_path);

    // Detect model type based on config files
    let model_type = detect_model_type(model_path)?;

    let (config, _) = match &model_type {
        ModelType::Base => {
            info!("Detected Base model type (no LoRA)");
            let config = load_base_model_config(model_path)?;
            (config, ())
        }
        ModelType::LoRA(lora_config) => {
            let config = load_base_model_config(model_path)?;
            info!("Detected Base model type with LoRA configuration:");
            info!("   • Alpha: {}", lora_config.lora_alpha);
            info!("   • Rank (r): {}", lora_config.r);
            info!("   • Target modules: {:?}", lora_config.target_modules);
            if let Some(ref base_model) = lora_config.base_model_name_or_path {
                info!("   • Base model: {}", base_model);
            }
            info!("");
            (config, ())
        }
    };

    Ok(ModelInfo { model_type, config })
}

/// Detect model type based on presence of config files.
/// For LoRA models, loads and embeds the LoRA configuration.
fn detect_model_type(model_path: &Path) -> Result<ModelType> {
    let has_adapter_config = model_path.join("adapter_config.json").exists();
    let has_base_config = model_path.join("config.json").exists();

    match (has_base_config, has_adapter_config) {
        (true, true) => {
            // LoRA model - load adapter config and embed the full config
            let lora_config = load_lora_config(model_path)?;
            Ok(ModelType::LoRA(lora_config))
        }
        (true, false) => Ok(ModelType::Base),
        (false, true) => anyhow::bail!(
            "Only LoRA config is found in {}. Make sure to have base model files in the same directory",
            model_path.display()
        ),
        _ => anyhow::bail!("No valid configuration files found in {}", model_path.display()),
    }
}

/// Load base model configuration - handles both direct config.json and LoRA case
fn load_base_model_config(model_path: &Path) -> Result<ModelConfig> {
    let config_path = model_path.join("config.json");

    if config_path.exists() {
        // Direct config.json exists
        load_hf_config(&config_path)
    } else {
        // For LoRA models, we might need to look elsewhere or use defaults
        // For now, return an error to let user know they need base model config
        anyhow::bail!(
            "Base model config.json not found in {}. For LoRA models, ensure the base model config is available.",
            model_path.display()
        )
    }
}

/// Load model configuration from HuggingFace format.
fn load_hf_config(config_path: &Path) -> Result<ModelConfig> {
    let mut file = File::open(config_path).with_context(|| format!("Failed to open config.json at {config_path:?}"))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let root: serde_json::Value =
        serde_json::from_str(&contents).map_err(|err| anyhow::anyhow!("Failed to parse config.json: {}", err))?;

    // Determine architecture from outer wrapper
    let architectures = root.get("architectures").and_then(|v| v.as_array());
    let architecture = match architectures {
        Some(arr) if arr.len() == 1 => {
            let arch_str = arr[0].as_str().ok_or_else(|| anyhow::anyhow!("architecture name is not a string"))?;
            ArchitectureId::try_from(arch_str)?
        }
        Some(arr) => anyhow::bail!("Multiple architectures are not supported: {arr:?}"),
        None => anyhow::bail!("Cannot determine architecture: missing 'architectures' field"),
    };

    // For nested wrapper architectures (Qwen3.5ForConditionalGeneration), extract text_config
    let config_src = if architecture == ArchitectureId::Qwen3_5ForConditionalGeneration {
        root.get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Missing 'text_config' field for Qwen3.5 architecture"))?
    } else {
        &root
    };

    #[derive(Debug, Deserialize)]
    struct RopeParameters {
        rope_theta: f32,
        partial_rotary_factor: f32,
    }

    #[derive(Debug, Deserialize)]
    struct FlatConfig {
        hidden_size: u32,
        intermediate_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        vocab_size: u32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        #[serde(default)]
        head_dim: Option<u32>,
        #[serde(default)]
        bos_token_id: Option<u32>,
        #[serde(default)]
        eos_token_id: Option<u32>,
        // Qwen3.5-specific fields (optional for Qwen3, required for Qwen3.5)
        #[serde(default)]
        hidden_act: Option<String>,
        #[serde(default)]
        attention_bias: Option<bool>,
        #[serde(default)]
        tie_word_embeddings: Option<bool>,
        #[serde(default)]
        layer_types: Option<Vec<String>>,
        // Correct JSON names for Qwen3.5 linear attention fields
        #[serde(default)]
        linear_num_key_heads: Option<u32>,
        #[serde(default)]
        linear_num_value_heads: Option<u32>,
        #[serde(default)]
        linear_key_head_dim: Option<u32>,
        #[serde(default)]
        linear_value_head_dim: Option<u32>,
        #[serde(default)]
        linear_conv_kernel_dim: Option<u32>,
        // rope_parameters is a nested object
        #[serde(default)]
        rope_parameters: Option<RopeParameters>,
    }

    let hf: FlatConfig = serde_json::from_value(config_src.clone())
        .map_err(|err| anyhow::anyhow!("Failed to parse model config: {err}"))?;

    let head_dim = hf.head_dim.unwrap_or(hf.hidden_size / hf.num_attention_heads);

    let mut config = ModelConfig {
        dim: hf.hidden_size,
        hidden_dim: hf.intermediate_size,
        n_layers: hf.num_hidden_layers,
        n_heads: hf.num_attention_heads,
        n_kv_heads: hf.num_key_value_heads,
        vocab_size: hf.vocab_size,
        max_seq_len: hf.max_position_embeddings,
        norm_eps: hf.rms_norm_eps,
        head_dim,
        bos_token_id: hf.bos_token_id.unwrap_or(0),
        eos_token_id: hf.eos_token_id.unwrap_or(0),
        architecture,
        qwen35: None,
    };

    // For Qwen3.5, validate and load extended config fields (required, no silent defaults)
    if architecture == ArchitectureId::Qwen3_5ForConditionalGeneration {
        // Required: hidden_act must be "silu"
        let hidden_act = hf
            .hidden_act
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'hidden_act'"))?;
        if hidden_act != "silu" {
            anyhow::bail!("Qwen3.5 requires hidden_act='silu', got '{hidden_act}'");
        }

        // Required: attention_bias must be false
        let attention_bias = hf
            .attention_bias
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'attention_bias'"))?;
        if attention_bias {
            anyhow::bail!("Qwen3.5 requires attention_bias=false, got true");
        }

        // Required: tie_word_embeddings must be true
        let tie_word_embeddings = hf
            .tie_word_embeddings
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'tie_word_embeddings'"))?;
        if !tie_word_embeddings {
            anyhow::bail!("Qwen3.5 requires tie_word_embeddings=true, got false");
        }

        // Required: layer_types with values "full_attention" and "linear_attention"
        let layer_types = hf
            .layer_types
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'layer_types'"))?;
        if hf.num_hidden_layers > 64 {
            anyhow::bail!("Qwen3.5 v2 supports at most 64 layers, got {}", hf.num_hidden_layers);
        }
        if layer_types.len() != hf.num_hidden_layers as usize {
            anyhow::bail!(
                "Qwen3.5 layer_types length {} does not match num_hidden_layers {}",
                layer_types.len(),
                hf.num_hidden_layers
            );
        }
        let mut full_attention_mask = 0u64;
        for (i, lt) in layer_types.iter().enumerate() {
            match lt.as_str() {
                "full_attention" => full_attention_mask |= 1u64 << i,
                "linear_attention" => {} // linear layers are NOT full-attention
                other => anyhow::bail!(
                    "Qwen3.5 unknown layer_type '{}' at index {}, expected 'full_attention' or 'linear_attention'",
                    other,
                    i
                ),
            }
        }

        // Required: linear_num_key_heads, linear_num_value_heads
        let n_linear_k_heads = hf
            .linear_num_key_heads
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'linear_num_key_heads'"))?;
        let n_linear_v_heads = hf
            .linear_num_value_heads
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'linear_num_value_heads'"))?;

        // Required: linear_key_head_dim, linear_value_head_dim
        let linear_k_head_dim = hf
            .linear_key_head_dim
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'linear_key_head_dim'"))?;
        let linear_v_head_dim = hf
            .linear_value_head_dim
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'linear_value_head_dim'"))?;

        // Required: linear_conv_kernel_dim
        let conv_kernel_size = hf
            .linear_conv_kernel_dim
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'linear_conv_kernel_dim'"))?;

        // Parse rope_parameters (nested object)
        let rope_params = hf
            .rope_parameters
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing required field 'rope_parameters'"))?;
        let rope_theta = rope_params.rope_theta;
        let computed_rotary = head_dim as f32 * rope_params.partial_rotary_factor;
        let rotary_dim = computed_rotary as u32;
        // Validate that rotary_dim is integral
        if (computed_rotary - rotary_dim as f32).abs() > 1e-6 {
            anyhow::bail!(
                "Qwen3.5 rotary_dim ({}) from head_dim={} * partial_rotary_factor={} is not integral",
                computed_rotary,
                head_dim,
                rope_params.partial_rotary_factor
            );
        }

        // Load tokenizer EOS from tokenizer_config.json
        let chat_eos = if let Some(tokenizer_eos) = load_chat_eos(config_path.parent().unwrap_or(config_path)) {
            tokenizer_eos
        } else {
            config.eos_token_id
        };

        config.qwen35 = Some(Qwen35Config {
            chat_eos,
            n_linear_k_heads,
            n_linear_v_heads,
            linear_k_head_dim,
            linear_v_head_dim,
            conv_kernel_size,
            rope_theta,
            rotary_dim,
            full_attention_mask,
        });
    }

    info!("Model configuration loaded:");
    info!("   • Architecture: {:?}", config.architecture);
    info!("   • Dimensions: {}", config.dim);
    info!("   • Layers: {}", config.n_layers);
    info!("   • Attention heads: {}", config.n_heads);
    info!("   • KV heads: {}", config.n_kv_heads);
    info!("   • Vocabulary size: {}", config.vocab_size);
    info!("   • Max sequence length: {}", config.max_seq_len);
    info!("   • Head dimension: {}", config.head_dim);
    info!("   • Model EOS: {}", config.eos_token_id);
    if let Some(ref q35) = config.qwen35 {
        info!("   • Chat EOS: {}", q35.chat_eos);
    }
    info!("");

    Ok(config)
}

/// Extract chat EOS token ID from tokenizer_config.json
fn load_chat_eos(model_dir: &Path) -> Option<u32> {
    let tokenizer_path = model_dir.join("tokenizer_config.json");
    if !tokenizer_path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&tokenizer_path).ok()?;
    let root: serde_json::Value = serde_json::from_str(&content).ok()?;

    // Try extracting eos_token value; it may be a string like "<|im_end|>" or a dict with "content"
    let eos_val = root.get("eos_token")?;
    let eos_str = match eos_val {
        serde_json::Value::String(s) => Some(s.as_str()),
        serde_json::Value::Object(m) => m.get("content").and_then(|v| v.as_str()),
        _ => None,
    }?;

    // Look up the token ID from added_tokens, added_tokens_decoder, or tokenizer.json

    // 1. Try added_tokens_decoder map (newer HuggingFace format, keyed by token ID string)
    if let Some(decoder) = root.get("added_tokens_decoder").and_then(|v| v.as_object()) {
        for (_id_str, entry) in decoder {
            if let Some(content) = entry.get("content").and_then(|v| v.as_str())
                && content == eos_str
                && let Ok(id) = _id_str.parse::<u64>()
            {
                return Some(id as u32);
            }
        }
    }

    // 2. Try added_tokens array (older format)
    if let Some(added) = root.get("added_tokens").and_then(|v| v.as_array()) {
        for entry in added {
            if let (Some(content), Some(id)) =
                (entry.get("content").and_then(|v| v.as_str()), entry.get("id").and_then(|v| v.as_u64()))
                && content == eos_str
            {
                return Some(id as u32);
            }
        }
    }

    // 3. Try tokenizer.json added_tokens as fallback
    let tok_path = model_dir.join("tokenizer.json");
    if let Ok(tok_content) = std::fs::read_to_string(&tok_path)
        && let Ok(tok_root) = serde_json::from_str::<serde_json::Value>(&tok_content)
        && let Some(added) = tok_root.get("added_tokens").and_then(|v| v.as_array())
    {
        for entry in added {
            if let (Some(content), Some(id)) =
                (entry.get("content").and_then(|v| v.as_str()), entry.get("id").and_then(|v| v.as_u64()))
                && content == eos_str
            {
                return Some(id as u32);
            }
        }
    }

    None
}

/// Load LoRA configuration from adapter_config.json
fn load_lora_config(model_path: &Path) -> Result<LoRAConfig> {
    let config_path = model_path.join("adapter_config.json");
    let mut file = File::open(&config_path)
        .with_context(|| format!("Failed to open adapter_config.json at {}", config_path.display()))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: LoRAConfig = serde_json::from_str(&contents)
        .map_err(|err| anyhow::anyhow!("Failed to parse adapter_config.json: {}", err))?;

    Ok(config)
}
