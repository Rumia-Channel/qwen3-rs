use super::*;
use crate::ModelConfig;

pub struct Qwen3_5 {
    weight_layers: Vec<WeightLayer<'static>>,
    header_v2: HeaderInfoV2,
    norm_layers: Vec<NormWeightLayer<'static>>,
}

impl Qwen3_5 {
    const ARCH_NAME: &'static str = "Qwen3_5ForConditionalGeneration";
    const EMBED_TOKENS_KEY: &'static str = "model.language_model.embed_tokens.weight";
    const LM_HEAD_KEY: &'static str = "lm_head.weight";

    pub fn new(config: ModelConfig) -> Self {
        let q35 = config.qwen35.as_ref().expect("Qwen3.5 config must contain qwen35 metadata");
        let mut weight_layers = Vec::new();

        for layer_idx in 0..config.n_layers {
            let prefix = format!("model.language_model.layers.{layer_idx}.");
            let is_full_attention = q35.full_attention_mask & (1_u64 << layer_idx) != 0;

            if is_full_attention {
                for component in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"] {
                    weight_layers.push(WeightLayer::new(format!("{prefix}{component}.weight"), component, layer_idx));
                }
            } else {
                for component in ["linear_attn.A_log", "linear_attn.dt_bias", "linear_attn.conv1d"] {
                    let suffix = if component == "linear_attn.conv1d" { ".weight" } else { "" };
                    weight_layers.push(WeightLayer::new_fp32(
                        format!("{prefix}{component}{suffix}"),
                        component,
                        layer_idx,
                    ));
                }
                for component in [
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.in_proj_b",
                    "linear_attn.in_proj_a",
                    "linear_attn.out_proj",
                ] {
                    weight_layers.push(WeightLayer::new(format!("{prefix}{component}.weight"), component, layer_idx));
                }
            }

            for component in ["mlp.gate_proj", "mlp.down_proj", "mlp.up_proj"] {
                weight_layers.push(WeightLayer::new(format!("{prefix}{component}.weight"), component, layer_idx));
            }
        }

        // Build dynamic norm layer list using config values
        let linear_v_head_dim = q35.linear_v_head_dim as usize;
        let head_dim = config.head_dim as usize;
        let norm_layers = vec![
            NormWeightLayer::new("model.language_model.layers.{}.input_layernorm.weight", true, true),
            NormWeightLayer::new("model.language_model.layers.{}.post_attention_layernorm.weight", true, true),
            NormWeightLayer::new("model.language_model.norm.weight", false, true),
            NormWeightLayer::new_with_dim(
                "model.language_model.layers.{}.self_attn.q_norm.weight",
                true,
                false,
                head_dim as u32,
            ),
            NormWeightLayer::new_with_dim(
                "model.language_model.layers.{}.self_attn.k_norm.weight",
                true,
                false,
                head_dim as u32,
            ),
            NormWeightLayer::new_with_dim(
                "model.language_model.layers.{}.linear_attn.norm.weight",
                true,
                false,
                linear_v_head_dim as u32,
            ),
        ];

        let header_v2 = HeaderInfoV2 {
            norm_eps: config.norm_eps,
            n_linear_k_heads: q35.n_linear_k_heads,
            n_linear_v_heads: q35.n_linear_v_heads,
            linear_k_head_dim: q35.linear_k_head_dim,
            linear_v_head_dim: q35.linear_v_head_dim,
            conv_kernel_size: q35.conv_kernel_size,
            rope_theta: q35.rope_theta,
            rotary_dim: q35.rotary_dim,
            full_attention_mask: q35.full_attention_mask,
            model_eos: config.eos_token_id,
            chat_eos: q35.chat_eos,
        };

        Self { weight_layers, header_v2, norm_layers }
    }
}

impl Architecture for Qwen3_5 {
    fn id(&self) -> ArchitectureId {
        ArchitectureId::Qwen3_5ForConditionalGeneration
    }

    fn name(&self) -> &'static str {
        Self::ARCH_NAME
    }

    fn header(&self) -> Result<HeaderInfo> {
        Ok(HeaderInfo { architecture_id: self.id() as u32, shared_classifier: true, v2: Some(self.header_v2.clone()) })
    }

    fn norm_weight_layers(&self) -> &[NormWeightLayer<'_>] {
        &self.norm_layers
    }

    fn embed_tokens_layer(&self) -> &'static str {
        Self::EMBED_TOKENS_KEY
    }

    fn lm_head_layer(&self) -> &'static str {
        Self::LM_HEAD_KEY
    }

    fn weight_layers(&self) -> &[WeightLayer<'_>] {
        &self.weight_layers
    }
}
