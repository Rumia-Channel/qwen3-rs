use std::io::Cursor;

use crate::utils::MemoryMapper;
use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};

/// Magic number for validating checkpoint files
const CHECKPOINT_MAGIC: i32 = 0x616a6331;
/// Size of the checkpoint header in bytes
const HEADER_SIZE: usize = 256;
/// Size of v1 config structure in bytes (13 × 4 = 52)
const CONFIG_V1_SIZE: usize = 52;
/// Size of v2 config structure in bytes (v1 + 12 fields = 100)
const CONFIG_V2_SIZE: usize = 100;

/// Configuration struct for transformer models.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture_id: usize,
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub vocab_size: usize,
    pub group_size: usize,
    pub shared_classifier: bool,
    /// v2 extended fields (zero / default for v1)
    pub v2: Option<ModelConfigV2>,
}

#[derive(Debug, Clone)]
pub struct ModelConfigV2 {
    pub norm_eps: f32,
    pub n_linear_k_heads: usize,
    pub n_linear_v_heads: usize,
    pub linear_k_head_dim: usize,
    pub linear_v_head_dim: usize,
    pub conv_kernel_size: usize,
    pub rope_theta: f32,
    pub rotary_dim: usize,
    pub full_attention_mask: u64,
    pub model_eos: u32,
    pub chat_eos: u32,
}

/// Reads and validates the model configuration from checkpoint data (mapper).
///
/// Supports version 1 (52-byte config) and version 2 (100-byte config).
pub fn read_config(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
    // Read magic + version (first 8 bytes)
    let magic_data = mapper.get_bytes(8)?;
    let mut cursor = Cursor::new(magic_data);
    let magic_number = cursor.read_i32::<LittleEndian>()?;
    let version = cursor.read_i32::<LittleEndian>()?;

    if magic_number != CHECKPOINT_MAGIC {
        anyhow::bail!("Invalid checkpoint magic number: expected {:#x}, got {:#x}", CHECKPOINT_MAGIC, magic_number);
    }

    match version {
        1 => read_config_v1(mapper),
        2 => read_config_v2(mapper),
        v => anyhow::bail!("Unsupported checkpoint version: expected 1 or 2, got {v}"),
    }
}

/// Read v1 header (52 bytes of config + padding).
fn read_config_v1(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
    // Remaining v1 fields after magic+version: 11 × 4 = 44 bytes
    let data = mapper.get_bytes(44)?;
    let mut cursor = Cursor::new(data);

    macro_rules! read_i32 {
        () => {
            cursor.read_i32::<LittleEndian>()?
        };
    }

    let architecture_id = read_i32!();
    let dim = read_i32!();
    let hidden_dim = read_i32!();
    let n_layers = read_i32!();
    let n_heads = read_i32!();
    let n_kv_heads = read_i32!();
    let vocab_size = read_i32!();
    let seq_len = read_i32!();
    let head_dim = read_i32!();
    let shared_classifier = read_i32!();
    let group_size = read_i32!();

    validate_positive_dims(&[
        ("architecture_id", architecture_id),
        ("dim", dim),
        ("n_layers", n_layers),
        ("n_heads", n_heads),
        ("n_kv_heads", n_kv_heads),
        ("vocab_size", vocab_size),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
    ])?;

    // Skip remaining header padding (256 - 52 = 204 bytes)
    mapper.skip(HEADER_SIZE - CONFIG_V1_SIZE)?;

    Ok(ModelConfig {
        architecture_id: architecture_id as usize,
        dim: dim as usize,
        hidden_dim: hidden_dim as usize,
        n_layers: n_layers as usize,
        n_heads: n_heads as usize,
        n_kv_heads: n_kv_heads as usize,
        head_dim: head_dim as usize,
        seq_len: seq_len as usize,
        vocab_size: vocab_size as usize,
        group_size: group_size as usize,
        shared_classifier: shared_classifier != 0,
        v2: None,
    })
}

/// Read v2 header (100 bytes of config + padding).
fn read_config_v2(mapper: &mut MemoryMapper) -> Result<ModelConfig> {
    // Remaining v2 fields after magic+version: v1 fields (44) + v2 extension (48) = 92 bytes
    let data = mapper.get_bytes(92)?;
    let mut cursor = Cursor::new(data);

    macro_rules! read_i32 {
        () => {
            cursor.read_i32::<LittleEndian>()?
        };
    }

    // ── v1 common fields (44 bytes) ──
    let architecture_id = read_i32!();
    let dim = read_i32!();
    let hidden_dim = read_i32!();
    let n_layers = read_i32!();
    let n_heads = read_i32!();
    let n_kv_heads = read_i32!();
    let vocab_size = read_i32!();
    let seq_len = read_i32!();
    let head_dim = read_i32!();
    let shared_classifier = read_i32!();
    let group_size = read_i32!();

    validate_positive_dims(&[
        ("architecture_id", architecture_id),
        ("dim", dim),
        ("n_layers", n_layers),
        ("n_heads", n_heads),
        ("n_kv_heads", n_kv_heads),
        ("vocab_size", vocab_size),
        ("seq_len", seq_len),
        ("head_dim", head_dim),
    ])?;

    // ── v2 extension fields (48 bytes) ──
    let norm_eps = cursor.read_f32::<LittleEndian>()?;
    let rope_theta = cursor.read_f32::<LittleEndian>()?;
    let rotary_dim = read_i32!();
    let n_linear_k_heads = read_i32!();
    let n_linear_v_heads = read_i32!();
    let linear_k_head_dim = read_i32!();
    let linear_v_head_dim = read_i32!();
    let conv_kernel_size = read_i32!();
    let full_attention_mask = cursor.read_u64::<LittleEndian>()?;
    let model_eos = cursor.read_u32::<LittleEndian>()?;
    let chat_eos = cursor.read_u32::<LittleEndian>()?;

    // Skip remaining header padding (256 - 100 = 156 bytes)
    mapper.skip(HEADER_SIZE - CONFIG_V2_SIZE)?;

    Ok(ModelConfig {
        architecture_id: architecture_id as usize,
        dim: dim as usize,
        hidden_dim: hidden_dim as usize,
        n_layers: n_layers as usize,
        n_heads: n_heads as usize,
        n_kv_heads: n_kv_heads as usize,
        head_dim: head_dim as usize,
        seq_len: seq_len as usize,
        vocab_size: vocab_size as usize,
        group_size: group_size as usize,
        shared_classifier: shared_classifier != 0,
        v2: Some(ModelConfigV2 {
            norm_eps,
            n_linear_k_heads: n_linear_k_heads as usize,
            n_linear_v_heads: n_linear_v_heads as usize,
            linear_k_head_dim: linear_k_head_dim as usize,
            linear_v_head_dim: linear_v_head_dim as usize,
            conv_kernel_size: conv_kernel_size as usize,
            rope_theta,
            rotary_dim: rotary_dim as usize,
            full_attention_mask,
            model_eos,
            chat_eos,
        }),
    })
}

fn validate_positive_dims(dims: &[(&str, i32)]) -> Result<()> {
    for (name, value) in dims {
        if *value <= 0 {
            anyhow::bail!("Invalid {}: must be positive, got {}", name, value);
        }
    }
    Ok(())
}
