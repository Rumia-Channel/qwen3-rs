use crate::configuration::ModelConfig;
use crate::models::Transformer;
use crate::sampler::Sampler;
use crate::tokenizer::Tokenizer;
use anyhow::Result;
use log::info;
use std::io::{self, Write};
use std::time::Instant;

pub fn generate<T: Transformer>(
    transformer: &mut T,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    prompt: Option<&str>,
) -> Result<()> {
    let prompt = prompt.unwrap_or("");
    let prompt_tokens = tokenizer.encode(prompt);

    if prompt_tokens.is_empty() {
        anyhow::bail!("Please provide a prompt");
    }

    let config = transformer.get_config().clone();
    let seq_len = config.seq_len;
    let prompt_len = prompt_tokens.len();

    if prompt_len > seq_len {
        anyhow::bail!("Prompt length ({}) exceeds context window ({})", prompt_len, seq_len);
    }

    // Reset cache before generation
    transformer.reset_cache();

    // Prefill all prompt tokens, compute logits only for the final token
    let logits = transformer.prefill(&prompt_tokens);
    if logits.is_empty() {
        anyhow::bail!("Prefill returned empty logits");
    }
    let mut logits_copy = logits.to_vec();
    let mut next_token = sampler.sample(&mut logits_copy);

    let mut pos = prompt_len;
    let mut metrics = TokenMetrics::new();
    metrics.start_generation();

    // Autoregressive decode for generated tokens
    while pos < seq_len {
        if is_termination_token(next_token, tokenizer, &config) {
            break;
        }

        output_token(tokenizer, next_token)?;
        metrics.increment_token();

        let logits = transformer.forward(next_token, pos);
        let mut logits_copy = logits.to_vec();
        next_token = sampler.sample(&mut logits_copy);
        pos += 1;
    }

    metrics.report_and_reset();
    println!();
    Ok(())
}

pub fn chat<T: Transformer>(
    transformer: &mut T,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    cli_user_prompt: Option<&str>,
    system_prompt: Option<&str>,
) -> Result<()> {
    let stdin = io::stdin();
    // Clone config to avoid borrow conflicts with &mut transformer calls.
    let config = transformer.get_config().clone();
    let seq_len = config.seq_len;
    let mut state = GenerationState::new();
    let mut user_turn = true;
    let mut next_token = 0;

    // Reset cache at the start of a chat session
    transformer.reset_cache();

    loop {
        // Reset context if window exceeded (context rollover)
        if state.pos >= seq_len {
            transformer.reset_cache();
            state.reset();
            user_turn = true;
            println!();
        }

        if user_turn {
            state.metrics.report_and_reset();

            if !handle_user_turn(
                &stdin,
                transformer,
                tokenizer,
                sampler,
                &mut state,
                &mut next_token,
                cli_user_prompt,
                system_prompt,
            )? {
                break;
            }
            user_turn = false;
        } else if handle_assistant_turn(
            transformer,
            tokenizer,
            sampler,
            &config,
            &mut state,
            &mut next_token,
            &mut user_turn,
        )? {
            continue; // Turn ended, continue to next iteration
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle_user_turn<T: Transformer>(
    stdin: &io::Stdin,
    transformer: &mut T,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    state: &mut GenerationState,
    next_token: &mut usize,
    cli_user_prompt: Option<&str>,
    system_prompt: Option<&str>,
) -> Result<bool> {
    let user_prompt = get_user_input(stdin, state.pos, cli_user_prompt)?;

    // Check if we should exit
    if user_prompt.is_empty() && !(state.pos == 0 && cli_user_prompt.is_some()) {
        return Ok(false);
    }

    let rendered_prompt = render_prompt(state.pos, system_prompt, &user_prompt, tokenizer);
    let prompt_tokens = tokenizer.encode(&rendered_prompt);

    if prompt_tokens.is_empty() {
        return Ok(true);
    }

    // Clamp to remaining context
    let available = transformer.get_config().seq_len.saturating_sub(state.pos);
    if available == 0 {
        return Ok(true);
    }
    let process_count = prompt_tokens.len().min(available);

    if state.pos == 0 {
        // First turn: use prefill for efficiency
        let tokens = &prompt_tokens[..process_count];
        let logits = transformer.prefill(tokens);
        if !logits.is_empty() {
            let mut logits_copy = logits.to_vec();
            *next_token = sampler.sample(&mut logits_copy);
        }
        state.pos = process_count;
    } else {
        // Subsequent turns: process token-by-token with forward
        for &token in prompt_tokens.iter().take(process_count) {
            *next_token = generate_next_token(transformer, sampler, token, state.pos)?;
            state.advance(token);
        }
    }

    Ok(true)
}

fn handle_assistant_turn<T: Transformer>(
    transformer: &mut T,
    tokenizer: &Tokenizer,
    sampler: &mut Sampler,
    config: &ModelConfig,
    state: &mut GenerationState,
    next_token: &mut usize,
    user_turn: &mut bool,
) -> Result<bool> {
    if is_termination_token(*next_token, tokenizer, config) {
        // Consume the stop token so the next user turn continues from the exact
        // chat transcript represented by the model cache.
        if state.pos < config.seq_len {
            transformer.forward(*next_token, state.pos);
            state.advance(*next_token);
        }
        state.metrics.report_and_reset();
        println!();
        *user_turn = true;
        return Ok(true);
    }

    state.metrics.start_generation();
    output_token(tokenizer, *next_token)?;

    *next_token = generate_next_token(transformer, sampler, *next_token, state.pos)?;
    state.metrics.increment_token();
    state.advance(*next_token);

    Ok(false)
}

fn generate_next_token<T: Transformer>(
    transformer: &mut T,
    sampler: &mut Sampler,
    token: usize,
    pos: usize,
) -> Result<usize> {
    let logits = transformer.forward(token, pos);
    let mut logits_copy = logits.to_vec();
    Ok(sampler.sample(&mut logits_copy))
}

fn output_token(tokenizer: &Tokenizer, token: usize) -> Result<()> {
    print!("{}", tokenizer.decode(token));
    io::stdout().flush()?;
    Ok(())
}

fn is_termination_token(token: usize, tokenizer: &Tokenizer, config: &ModelConfig) -> bool {
    // Always stop on tokenizer EOS
    if token == tokenizer.eos_token_id as usize {
        return true;
    }
    // v2 model-specific EOS tokens (model_eos, chat_eos)
    if let Some(v2) = &config.v2
        && (token == v2.model_eos as usize || token == v2.chat_eos as usize)
    {
        return true;
    }
    // Legacy: stop on BOS only for v1 models (Qwen3).
    // For v2 models (Qwen3.5), BOS=0 is a valid token and must not stop generation.
    if config.v2.is_none() && token == tokenizer.bos_token_id as usize {
        return true;
    }
    false
}

fn get_user_input(stdin: &io::Stdin, pos: usize, cli_user_prompt: Option<&str>) -> Result<String> {
    match (pos, cli_user_prompt) {
        (0, Some(prompt)) => Ok(prompt.to_string()),
        (_, Some(_)) => Ok(String::new()), // Signal to break (single-prompt chat mode)
        _ => {
            print!("> ");
            io::stdout().flush()?;
            let mut input = String::new();
            stdin.read_line(&mut input)?;
            Ok(input.trim().to_string())
        }
    }
}

fn render_prompt(pos: usize, system_prompt: Option<&str>, user_prompt: &str, tokenizer: &Tokenizer) -> String {
    match (pos, system_prompt) {
        (0, Some(sys_prompt)) => {
            tokenizer.system_prompt_template.replace("%s", &format!("{sys_prompt}\n{user_prompt}"))
        }
        _ => tokenizer.prompt_template.replace("%s", user_prompt),
    }
}

/// Tracks token generation performance metrics
struct TokenMetrics {
    start_time: Option<Instant>,
    generated_count: usize,
}

impl TokenMetrics {
    fn new() -> Self {
        Self { start_time: None, generated_count: 0 }
    }

    fn start_generation(&mut self) {
        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }
    }

    fn increment_token(&mut self) {
        self.generated_count += 1;
    }

    fn report_and_reset(&mut self) {
        if let Some(start_time) = self.start_time.take() {
            let duration = start_time.elapsed();
            if self.generated_count > 0 && duration.as_secs_f64() > 0.0 {
                let tps = self.generated_count as f64 / duration.as_secs_f64();
                info!(
                    "\n[Generated {} tokens in {:.2}s - {:.2} tokens/sec]",
                    self.generated_count,
                    duration.as_secs_f64(),
                    tps
                );
            }
        }
        self.generated_count = 0;
    }
}

/// Represents the current generation state
struct GenerationState {
    pos: usize,
    metrics: TokenMetrics,
}

impl GenerationState {
    fn new() -> Self {
        Self { pos: 0, metrics: TokenMetrics::new() }
    }

    fn reset(&mut self) {
        self.metrics.report_and_reset();
        self.pos = 0;
    }

    fn advance(&mut self, _next_token: usize) {
        self.pos += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::configuration::ModelConfigV2;

    /// Helper: minimal v1 config (no v2 fields).
    fn v1_config() -> ModelConfig {
        ModelConfig {
            architecture_id: 1,
            dim: 64,
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            seq_len: 128,
            vocab_size: 1000,
            group_size: 32,
            shared_classifier: true,
            v2: None,
        }
    }

    /// Helper: minimal v2 config (Qwen3.5-style).
    fn v2_config() -> ModelConfig {
        ModelConfig {
            architecture_id: 3,
            dim: 64,
            hidden_dim: 256,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 16,
            seq_len: 128,
            vocab_size: 1000,
            group_size: 32,
            shared_classifier: true,
            v2: Some(ModelConfigV2 {
                norm_eps: 1e-6,
                n_linear_k_heads: 4,
                n_linear_v_heads: 4,
                linear_k_head_dim: 16,
                linear_v_head_dim: 16,
                conv_kernel_size: 4,
                rope_theta: 10000000.0,
                rotary_dim: 64,
                full_attention_mask: 0b0000100010001000,
                model_eos: 999,
                chat_eos: 998,
            }),
        }
    }

    /// Minimal tokenizer with a non-zero BOS.
    fn tokenizer_v1() -> Tokenizer {
        Tokenizer {
            vocab: vec![b"dummy".to_vec()],
            merge_scores: vec![0.0],
            vocab_size: 1,
            max_token_length: 10,
            bos_token_id: 1,
            eos_token_id: 2,
            prompt_template: String::new(),
            system_prompt_template: String::new(),
        }
    }

    /// Minimal tokenizer with BOS=0 (Qwen3.5-style, where 0 is a real token).
    fn tokenizer_v2() -> Tokenizer {
        Tokenizer {
            vocab: vec![b"valid".to_vec(), b"thing".to_vec()],
            merge_scores: vec![0.0, 0.0],
            vocab_size: 2,
            max_token_length: 10,
            bos_token_id: 0,
            eos_token_id: 2,
            prompt_template: String::new(),
            system_prompt_template: String::new(),
        }
    }

    // ── is_termination_token tests ──

    #[test]
    fn test_v1_stops_on_eos() {
        let config = v1_config();
        let tok = tokenizer_v1();
        // eos = 2 should stop
        assert!(is_termination_token(2, &tok, &config));
    }

    #[test]
    fn test_v1_stops_on_bos() {
        let config = v1_config();
        let tok = tokenizer_v1();
        // bos = 1 should stop for v1 (legacy behavior)
        assert!(is_termination_token(1, &tok, &config));
    }

    #[test]
    fn test_v1_non_terminal_passes() {
        let config = v1_config();
        let tok = tokenizer_v1();
        // token 42 not in {bos=1, eos=2}
        assert!(!is_termination_token(42, &tok, &config));
    }

    #[test]
    fn test_v2_stops_on_eos() {
        let config = v2_config();
        let tok = tokenizer_v2();
        // eos = 2 should stop
        assert!(is_termination_token(2, &tok, &config));
    }

    #[test]
    fn test_v2_stops_on_model_eos() {
        let config = v2_config();
        let tok = tokenizer_v2();
        // model_eos = 999 should stop
        assert!(is_termination_token(999, &tok, &config));
    }

    #[test]
    fn test_v2_stops_on_chat_eos() {
        let config = v2_config();
        let tok = tokenizer_v2();
        // chat_eos = 998 should stop
        assert!(is_termination_token(998, &tok, &config));
    }

    #[test]
    fn test_v2_does_not_stop_on_bos_zero() {
        let config = v2_config();
        let tok = tokenizer_v2();
        // bos = 0 is a valid token in v2, must NOT stop
        assert!(!is_termination_token(0, &tok, &config));
    }

    #[test]
    fn test_v2_non_terminal_passes() {
        let config = v2_config();
        let tok = tokenizer_v2();
        // token 777 not in any stop set
        assert!(!is_termination_token(777, &tok, &config));
    }

    #[test]
    fn test_v2_also_stops_on_bos_when_nonzero() {
        // Edge: if a hypothetical v2 model had BOS=1 (non-zero),
        // it should still NOT stop because the BOS check is v1-only.
        // This test documents the deliberate choice.
        let config = v2_config();
        let mut tok = tokenizer_v2();
        tok.bos_token_id = 1;
        assert!(!is_termination_token(1, &tok, &config));
    }
}
