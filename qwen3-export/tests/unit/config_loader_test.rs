//! Integration tests for config loader functionality

use super::*;
use anyhow::Result;
use std::{fs, path::PathBuf};
use tempfile::TempDir;

/// Helper to create a minimal config.json for testing
fn create_test_config_json(temp_dir: &TempDir) -> Result<PathBuf> {
    let config_content = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6,
        "head_dim": 32,
        "bos_token_id": 1,
        "eos_token_id": 2
    }"#;
    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path.clone(), config_content)?;

    Ok(config_path)
}

#[test]
fn test_load_hf_config_valid() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let path = create_test_config_json(&temp_dir)?;

    let config = load_hf_config(&path)?;

    // Verify all fields are loaded correctly
    assert_eq!(config.dim, 256);
    assert_eq!(config.hidden_dim, 1024);
    assert_eq!(config.n_layers, 4);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.n_kv_heads, 8);
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.max_seq_len, 512);
    assert_eq!(config.head_dim, 32);
    assert!((config.norm_eps - 1e-6).abs() < 1e-9);
    assert_eq!(config.bos_token_id, 1);
    assert_eq!(config.eos_token_id, 2);

    Ok(())
}

#[test]
fn test_load_hf_config_invalid_json() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path.clone(), "invalid json")?;

    let result = load_hf_config(&config_path);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "Failed to parse config.json: expected value at line 1 column 1");

    Ok(())
}

#[test]
fn test_load_hf_config_missing_required_field() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Config missing both architecture and hidden_size
    let config_content = r#"{
        "intermediate_size": 1024,
        "num_hidden_layers": 4
    }"#;

    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path.clone(), config_content)?;

    let result = load_hf_config(&config_path);
    assert!(result.is_err());
    // The new loader first checks for the architecture field
    assert!(result.unwrap_err().to_string().contains("architectures"));

    Ok(())
}

#[test]
fn test_load_official_qwen35_2b_config() -> Result<()> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sample/Qwen/Qwen3.5-2B/config.json");
    let config = load_hf_config(&path)?;

    assert_eq!(config.architecture, ArchitectureId::Qwen3_5ForConditionalGeneration);
    assert_eq!(config.dim, 2048);
    assert_eq!(config.hidden_dim, 6144);
    assert_eq!(config.n_layers, 24);
    assert_eq!(config.n_heads, 8);
    assert_eq!(config.n_kv_heads, 2);
    assert_eq!(config.head_dim, 256);
    assert_eq!(config.vocab_size, 248_320);
    assert_eq!(config.max_seq_len, 262_144);
    assert_eq!(config.eos_token_id, 248_044);

    let q35 = config.qwen35.expect("Qwen3.5 metadata");
    assert_eq!(q35.chat_eos, 248_046);
    assert_eq!(q35.n_linear_k_heads, 16);
    assert_eq!(q35.n_linear_v_heads, 16);
    assert_eq!(q35.linear_k_head_dim, 128);
    assert_eq!(q35.linear_v_head_dim, 128);
    assert_eq!(q35.conv_kernel_size, 4);
    assert_eq!(q35.rotary_dim, 64);
    assert_eq!(q35.full_attention_mask, 0x88_88_88);
    assert_eq!(q35.rope_theta, 10_000_000.0);

    Ok(())
}

#[test]
fn test_load_hf_config_with_defaults() -> Result<()> {
    let temp_dir = TempDir::new()?;

    // Config without optional fields (bos_token_id, eos_token_id, head_dim)
    let config_content = r#"{
        "architectures": ["Qwen3ForCausalLM"],
        "hidden_size": 256,
        "intermediate_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "vocab_size": 1000,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6
    }"#;

    let config_path = temp_dir.path().join("config.json");
    fs::write(config_path.clone(), config_content)?;

    let config = load_hf_config(&config_path)?;

    // Check defaults are applied
    assert_eq!(config.bos_token_id, 0); // default
    assert_eq!(config.eos_token_id, 0); // default
    assert_eq!(config.head_dim, 256 / 8); // calculated: dim / n_heads

    Ok(())
}
