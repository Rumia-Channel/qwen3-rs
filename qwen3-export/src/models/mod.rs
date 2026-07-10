use anyhow::Result;

use crate::{ModelInfo, models::qwen3::Qwen3, models::qwen3_5::Qwen3_5, tensor_reader::TensorReader};

mod qwen3;
mod qwen3_5;

/// Architecture ID for binary format identification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ArchitectureId {
    Qwen3ForCausalLM = 1,
    LlamaForCausalLM = 2,
    Qwen3_5ForConditionalGeneration = 3,
}

impl TryFrom<&str> for ArchitectureId {
    type Error = anyhow::Error;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "Qwen3ForCausalLM" => Ok(Self::Qwen3ForCausalLM),
            "LlamaForCausalLM" => Ok(Self::LlamaForCausalLM),
            "Qwen3_5ForConditionalGeneration" => Ok(Self::Qwen3_5ForConditionalGeneration),
            _ => anyhow::bail!("Unknown ArchitectureId: {value}"),
        }
    }
}

impl TryFrom<u32> for ArchitectureId {
    type Error = anyhow::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            1 => Ok(Self::Qwen3ForCausalLM),
            2 => Ok(Self::LlamaForCausalLM),
            3 => Ok(Self::Qwen3_5ForConditionalGeneration),
            _ => anyhow::bail!("Unknown ArchitectureId: {value}"),
        }
    }
}

/// Header information: v1 fields plus optional v2 extension.
#[derive(Debug, Clone)]
pub struct HeaderInfo {
    pub architecture_id: u32,
    pub shared_classifier: bool,
    /// v2 extended fields (ignored / zero for v1 architectures).
    pub v2: Option<HeaderInfoV2>,
}

#[derive(Debug, Clone)]
pub struct HeaderInfoV2 {
    pub norm_eps: f32,
    pub n_linear_k_heads: u32,
    pub n_linear_v_heads: u32,
    pub linear_k_head_dim: u32,
    pub linear_v_head_dim: u32,
    pub conv_kernel_size: u32,
    pub rope_theta: f32,
    pub rotary_dim: u32,
    pub full_attention_mask: u64,
    pub model_eos: u32,
    pub chat_eos: u32,
}

/// Represents normalization layer.
pub struct NormWeightLayer<'a> {
    /// Name of the layer
    pub name: &'a str,
    /// If set to true, name is a pattern parametrized with layer index
    pub layered: bool,
    /// If true, error will be returned if the layer not found
    /// Otherwise, default(1.0) value will be set.
    pub is_required: bool,
    /// Dimension override for default values when weight is missing (0 = use head_dim).
    pub default_dim: u32,
}

impl<'a> NormWeightLayer<'a> {
    pub const fn new(pattern: &'a str, layered: bool, is_required: bool) -> Self {
        Self { name: pattern, layered, is_required, default_dim: 0 }
    }

    pub const fn new_with_dim(pattern: &'a str, layered: bool, is_required: bool, default_dim: u32) -> Self {
        Self { name: pattern, layered, is_required, default_dim }
    }
}

pub struct WeightLayer<'a> {
    pub tensor_name: String,
    pub component: &'a str,
    pub layer_idx: u32,
    /// If true, this weight is kept in FP32; if false, quantised to Q8.
    pub is_fp32: bool,
}

impl<'a> WeightLayer<'a> {
    pub fn new(tensor_name: String, component: &'a str, layer_idx: u32) -> Self {
        Self { tensor_name, component, layer_idx, is_fp32: false }
    }

    pub fn new_fp32(tensor_name: String, component: &'a str, layer_idx: u32) -> Self {
        Self { tensor_name, component, layer_idx, is_fp32: true }
    }
}

pub trait Architecture {
    fn id(&self) -> ArchitectureId;

    fn name(&self) -> &'static str;

    fn header(&self) -> Result<HeaderInfo>;

    fn norm_weight_layers(&self) -> &[NormWeightLayer<'_>];

    fn embed_tokens_layer(&self) -> &'static str;

    fn lm_head_layer(&self) -> &'static str;

    fn weight_layers(&self) -> &[WeightLayer<'_>];
}

pub fn create_architecture<'a>(model_info: &ModelInfo, tensor_reader: &'a TensorReader) -> Box<dyn Architecture + 'a> {
    match model_info.config.architecture {
        ArchitectureId::Qwen3ForCausalLM => Box::new(Qwen3::new(model_info, tensor_reader)),
        ArchitectureId::Qwen3_5ForConditionalGeneration => Box::new(Qwen3_5::new(model_info.config.clone())),
        ArchitectureId::LlamaForCausalLM => todo!("LlamaForCausalLM not yet implemented"),
    }
}
