//! Qwen3.5 dense text inference with hybrid GatedFullAttention / GatedDeltaNet layers.
//!
//! Architecture (per design.md):
//! - 24 layers: layers 3,7,11,15,19,23 are GatedFullAttention, rest are GatedDeltaNet.
//! - Standard RMSNorm (1+w) for input/post/Q/K norms.
//! - Gated RMSNorm (rms(x)*w*silu(z)) for DeltaNet norm.
//! - Partial RoPE on first rotary_dim dimensions, rotate-half pairing.
//! - FP32 Gated Delta recurrence with per-head state S [kd×vd].
//! - Depthwise causal conv1d width 4, chronological ring ordering.
//! - Selected-row embedding dequantization.

use crate::configuration::ModelConfig;
use crate::tensor::{QuantizedTensor, matmul, quantize};
use crate::utils::MemoryMapper;
use anyhow::{Context, Result, anyhow, ensure};

// ---------------------------------------------------------------------------
// Standard RMSNorm: rms(x) * (1 + w)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StandardRMSNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl StandardRMSNorm {
    pub fn new(weight: Vec<f32>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Apply norm in-place.
    pub fn forward_inplace(&self, x: &mut [f32]) {
        let len = x.len();
        let sum_sq = x.iter().map(|&v| v * v).sum::<f32>();
        let rms = (sum_sq / len as f32 + self.eps).sqrt().recip();
        for (val, &w) in x.iter_mut().zip(self.weight.iter()) {
            *val = rms * *val * (1.0 + w);
        }
    }

    /// Apply norm to `input`, write into `output`.
    pub fn forward(&self, output: &mut [f32], input: &[f32]) {
        let len = input.len();
        let sum_sq = input.iter().map(|&v| v * v).sum::<f32>();
        let rms = (sum_sq / len as f32 + self.eps).sqrt().recip();
        for (out, (&inp, &w)) in output.iter_mut().zip(input.iter().zip(self.weight.iter())) {
            *out = rms * inp * (1.0 + w);
        }
    }
}

// ---------------------------------------------------------------------------
// Gated RMSNorm: per value head, weight length = linear_v_head_dim
//   For each value head h: x_h = rms(x_h) * w * silu(gate_h)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct GatedRMSNorm {
    weight: Vec<f32>,
    eps: f32,
    n_v_heads: usize,
    v_head_dim: usize,
}

impl GatedRMSNorm {
    pub fn new(weight: Vec<f32>, eps: f32, n_v_heads: usize, v_head_dim: usize) -> Self {
        Self { weight, eps, n_v_heads, v_head_dim }
    }

    /// Apply gated norm per value head: x[h] *= w * silu(gate[h])
    /// `x` and `gate` each have length n_v_heads * v_head_dim.
    pub fn forward(&self, x: &mut [f32], gate: &[f32]) {
        let vd = self.v_head_dim;
        let nh = self.n_v_heads;
        for h in 0..nh {
            let start = h * vd;
            let head_x = &mut x[start..start + vd];
            let head_gate = &gate[start..start + vd];
            let sum_sq = head_x.iter().map(|&v| v * v).sum::<f32>();
            let rms = (sum_sq / vd as f32 + self.eps).sqrt().recip();
            for i in 0..vd {
                let silu_g = head_gate[i] * (1.0 + (-head_gate[i]).exp()).recip();
                head_x[i] = rms * head_x[i] * self.weight[i] * silu_g;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Partial RoPE with rotate-half pairing
// ---------------------------------------------------------------------------

pub fn apply_partial_rope(slice: &mut [f32], pos: usize, rotary_dim: usize, theta: f32) {
    let half = rotary_dim.min(slice.len()) / 2;
    for i in 0..half {
        let inv_freq = theta.powf(-(2.0 * i as f32) / rotary_dim as f32);
        let angle = pos as f32 * inv_freq;
        let (cos, sin) = (angle.cos(), angle.sin());
        // rotate-half: pair i with i + half
        let a = slice[i];
        let b = slice[i + half];
        slice[i] = a * cos - b * sin;
        slice[i + half] = a * sin + b * cos;
    }
}

// ---------------------------------------------------------------------------
// L2 normalize and scale
// ---------------------------------------------------------------------------

fn l2_normalize(x: &mut [f32], eps: f32) {
    let sum_sq = x.iter().map(|&v| v * v).sum::<f32>();
    let inv_norm = (sum_sq + eps).sqrt().recip();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

// ---------------------------------------------------------------------------
// Quantized linear wrapper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct QLinear {
    weight: QuantizedTensor,
    in_features: usize,
    out_features: usize,
    group_size: usize,
}

impl QLinear {
    fn forward(&self, output: &mut [f32], input: &QuantizedTensor) {
        matmul(output, input, &self.weight, self.in_features, self.out_features, self.group_size);
    }
}

// ---------------------------------------------------------------------------
// Gated Full Attention
// ---------------------------------------------------------------------------

pub(in crate::models) struct GatedFullAttention {
    pub(crate) wq: QLinear,
    pub(crate) wk: QLinear,
    pub(crate) wv: QLinear,
    pub(crate) wo: QLinear,
    pub(crate) q_norm: StandardRMSNorm,
    pub(crate) k_norm: StandardRMSNorm,
    pub(crate) n_heads: usize,
    pub(crate) n_kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) kv_mul: usize,
    pub(crate) rotary_dim: usize,
    pub(crate) rope_theta: f32,
    pub(crate) group_size: usize,
}

impl GatedFullAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xq: &QuantizedTensor,
        key_cache: &mut [f32],
        value_cache: &mut [f32],
        q_buf: &mut [f32],   // [2 * n_heads * head_dim]
        k_buf: &mut [f32],   // [kv_dim]
        v_buf: &mut [f32],   // [kv_dim]
        att_buf: &mut [f32], // [n_heads * seq_len]
        output: &mut [f32],  // [all_heads_dim]
        pos: usize,
        kv_offset: usize,
        seq_len: usize,
    ) {
        let n_heads = self.n_heads;
        let n_kv = self.n_kv_heads;
        let hd = self.head_dim;
        let kv_hd = n_kv * hd;
        let all_hd = n_heads * hd;

        // Q projection outputs 2 * all_heads_dim: per head [Q(256), gate(256)] interleaved
        self.wq.forward(q_buf, xq);

        // Deinterleave q_buf into q and gate
        // q_buf layout: [Q_head0(256), gate_head0(256), Q_head1(256), gate_head1(256), ...]
        let mut q = vec![0.0; all_hd];
        let mut gate = vec![0.0; all_hd];
        for h in 0..n_heads {
            let src_off = h * 2 * hd;
            let dst_off = h * hd;
            q[dst_off..dst_off + hd].copy_from_slice(&q_buf[src_off..src_off + hd]);
            gate[dst_off..dst_off + hd].copy_from_slice(&q_buf[src_off + hd..src_off + 2 * hd]);
        }

        // K/V projections
        self.wk.forward(k_buf, xq);
        self.wv.forward(v_buf, xq);

        // Store K/V in cache
        key_cache[kv_offset..kv_offset + kv_hd].copy_from_slice(&k_buf[..kv_hd]);
        value_cache[kv_offset..kv_offset + kv_hd].copy_from_slice(&v_buf[..kv_hd]);

        // Q norm + partial RoPE
        for h in 0..n_heads {
            let start = h * hd;
            let q_slice = &mut q[start..start + hd];
            let mut tmp = vec![0.0; hd];
            tmp.copy_from_slice(q_slice);
            self.q_norm.forward(q_slice, &tmp);
            apply_partial_rope(q_slice, pos, self.rotary_dim, self.rope_theta);
        }

        // K norm + partial RoPE (in-place on cache)
        for h in 0..n_kv {
            let start = kv_offset + h * hd;
            let k_slice = &mut key_cache[start..start + hd];
            let mut tmp = vec![0.0; hd];
            tmp.copy_from_slice(k_slice);
            self.k_norm.forward(k_slice, &tmp);
            apply_partial_rope(k_slice, pos, self.rotary_dim, self.rope_theta);
        }

        // GQA attention
        let inv_sqrt_d = (hd as f32).sqrt().recip();
        let kv_base = kv_offset - pos * kv_hd;

        for h in 0..n_heads {
            let kv_h = h / self.kv_mul;
            let q_slice = &q[h * hd..(h + 1) * hd];
            let att_scores = &mut att_buf[h * seq_len..h * seq_len + pos + 1];

            // Scores
            for (t, att_score) in att_scores.iter_mut().enumerate() {
                let k_base = kv_base + t * kv_hd + kv_h * hd;
                let score = q_slice.iter().zip(&key_cache[k_base..k_base + hd]).map(|(&q, &k)| q * k).sum::<f32>();
                *att_score = score * inv_sqrt_d;
            }

            // Softmax
            let max_val = att_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f32 = att_scores
                .iter_mut()
                .map(|s| {
                    *s = (*s - max_val).exp();
                    *s
                })
                .sum();
            let inv_sum = sum_exp.recip();
            for s in att_scores.iter_mut() {
                *s *= inv_sum;
            }

            // Weighted sum of V
            let out_slice = &mut output[h * hd..(h + 1) * hd];
            out_slice.fill(0.0);
            for (t, &weight) in att_scores.iter().enumerate() {
                let v_base = kv_base + t * kv_hd + kv_h * hd;
                out_slice
                    .iter_mut()
                    .zip(&value_cache[v_base..v_base + hd])
                    .for_each(|(out, &value)| *out += weight * value);
            }
        }

        // Output gate (sigmoid)
        for i in 0..all_hd {
            let g = (1.0 + (-gate[i]).exp()).recip();
            output[i] *= g;
        }

        // Output projection
        let mut out_q = QuantizedTensor::new(all_hd, self.group_size);
        quantize(&mut out_q, output, all_hd, self.group_size);
        self.wo.forward(output, &out_q);
    }
}

// ---------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------

pub(in crate::models) struct GatedDeltaNet {
    pub(crate) w_qkv: QLinear,
    pub(crate) w_z: QLinear,
    pub(crate) w_a: QLinear,
    pub(crate) w_b: QLinear,
    pub(crate) w_out: QLinear,
    pub(crate) a_log: Vec<f32>,
    pub(crate) dt_bias: Vec<f32>,
    pub(crate) conv_weight: Vec<f32>,
    pub(crate) gated_norm: GatedRMSNorm,
    pub(crate) n_k_heads: usize,
    pub(crate) k_head_dim: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) q_head_dim: usize,
    pub(crate) qkv_dim: usize,
    pub(crate) conv_kernel: usize,
    pub(crate) group_size: usize,
    pub(crate) dim: usize,
    pub(crate) norm_eps: f32,
}

impl GatedDeltaNet {
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        xq: &QuantizedTensor,
        z_buf: &mut [f32],
        qkv_buf: &mut [f32],
        a_buf: &mut [f32],
        b_buf: &mut [f32],
        conv_state_out: &mut [f32],
        conv_buf: &mut [f32],        // [qkv_dim * kernel_size]
        recurrent_state: &mut [f32], // [n_heads * kd * vd] (n_heads = n_kh = n_vh)
        output: &mut [f32],
        pos: usize,
    ) {
        let n_heads = self.n_k_heads; // == n_v_heads (validated equal in constructor)
        let kd = self.k_head_dim;
        let vd = self.v_head_dim;
        let qd = self.q_head_dim;
        let qkv_dim = self.qkv_dim;
        let ks = self.conv_kernel;
        let eps = self.norm_eps;

        // Projections
        self.w_qkv.forward(qkv_buf, xq);
        self.w_z.forward(z_buf, xq);
        self.w_a.forward(a_buf, xq);
        self.w_b.forward(b_buf, xq);

        // Depthwise causal convolution with chronological ring
        let stride = qkv_dim;
        let write_slot = (pos % ks) * stride;
        conv_buf[write_slot..write_slot + stride].copy_from_slice(qkv_buf);

        for (ch, conv_out) in conv_state_out.iter_mut().enumerate().take(qkv_dim) {
            *conv_out = (0..ks)
                .map(|t| {
                    let read_slot = ((pos + 1 + t) % ks) * stride + ch;
                    self.conv_weight[ch * ks + t] * conv_buf[read_slot]
                })
                .sum();
        }
        // SiLU
        for val in conv_state_out.iter_mut() {
            *val = *val * (1.0 + (-*val).exp()).recip();
        }

        // Split Q, K, V (1:1 head pairing)
        let q_part = &conv_state_out[..n_heads * qd];
        let k_part = &conv_state_out[n_heads * qd..n_heads * qd + n_heads * kd];
        let v_part = &conv_state_out[n_heads * qd + n_heads * kd..n_heads * qd + n_heads * kd + n_heads * vd];

        // Per-head state: [n_heads, kd, vd]
        let state_size = kd * vd;
        let inv_sqrt_kd = (kd as f32).sqrt().recip();

        let mut q_normalized = vec![0.0; n_heads * qd];

        for h in 0..n_heads {
            // Q: normalize and scale
            let q_start = h * qd;
            let q_slice = &mut q_normalized[q_start..q_start + qd];
            q_slice.copy_from_slice(&q_part[q_start..q_start + qd]);
            l2_normalize(q_slice, eps);
            for v in q_slice.iter_mut() {
                *v *= inv_sqrt_kd;
            }

            // K: normalize
            let k_start = h * kd;
            let mut k_norm = k_part[k_start..k_start + kd].to_vec();
            l2_normalize(&mut k_norm, eps);

            // V
            let v_raw = &v_part[h * vd..(h + 1) * vd];

            // Per-head scalar a/b (sized for n_linear_v_heads == n_heads)
            let beta = (1.0 + (-b_buf[h]).exp()).recip();
            let x = a_buf[h] + self.dt_bias[h];
            let sp = if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() };
            let g_val = -self.a_log[h].exp() * sp;
            let decay = g_val.exp();

            let s_off = h * state_size;

            // Decay S *= exp(g)
            for i in 0..state_size {
                recurrent_state[s_off + i] *= decay;
            }

            // predicted_v = S^T @ k_norm, delta = (v - predicted) * beta, S += k_norm outer delta
            for j in 0..vd {
                let mut predicted = 0.0;
                for i in 0..kd {
                    predicted += recurrent_state[s_off + i * vd + j] * k_norm[i];
                }
                let delta = (v_raw[j] - predicted) * beta;
                for i in 0..kd {
                    recurrent_state[s_off + i * vd + j] += k_norm[i] * delta;
                }
            }
        }

        // Output: S^T @ q_normalized (produce n_heads * vd)
        for h in 0..n_heads {
            let q_start = h * qd;
            let s_off = h * state_size;
            let out_off = h * vd;
            for j in 0..vd {
                let mut out_val = 0.0;
                for i in 0..kd {
                    out_val += recurrent_state[s_off + i * vd + j] * q_normalized[q_start + i];
                }
                output[out_off + j] = out_val;
            }
        }

        // Gated norm per value head
        self.gated_norm.forward(output, z_buf);

        // Output projection
        let mut out_q = QuantizedTensor::new(self.dim, self.group_size);
        quantize(&mut out_q, output, self.dim, self.group_size);
        self.w_out.forward(output, &out_q);
    }
}

// ---------------------------------------------------------------------------
// Per-layer containers
// ---------------------------------------------------------------------------

struct FullAttnLayer {
    attn: GatedFullAttention,
    mlp_w1: QLinear,
    mlp_w2: QLinear,
    mlp_w3: QLinear,
}

struct DeltaLayer {
    delta: GatedDeltaNet,
    mlp_w1: QLinear,
    mlp_w2: QLinear,
    mlp_w3: QLinear,
}

enum LayerWeights {
    FullAttention(FullAttnLayer),
    DeltaNet(DeltaLayer),
}

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

struct TransformerBuffers {
    x: Vec<f32>,         // residual stream [dim]
    x_normed: Vec<f32>,  // normalized scratch for mixer [dim]
    xb_normed: Vec<f32>, // normalized scratch for MLP [dim]
    xq: QuantizedTensor,
    hb: Vec<f32>,
    hb2: Vec<f32>,
    hq: QuantizedTensor,
    q_buf: Vec<f32>,    // [2 * n_heads * head_dim]
    k_buf: Vec<f32>,    // [kv_dim]
    v_buf: Vec<f32>,    // [kv_dim]
    att_buf: Vec<f32>,  // [n_heads * seq_len]
    z_buf: Vec<f32>,    // [dim]
    qkv_buf: Vec<f32>,  // [qkv_dim]
    a_buf: Vec<f32>,    // [n_linear_v_heads]
    b_buf: Vec<f32>,    // [n_linear_v_heads]
    conv_out: Vec<f32>, // [qkv_dim]
    ffn_out: Vec<f32>,  // [dim]
}

impl TransformerBuffers {
    #[allow(clippy::too_many_arguments)]
    fn new(
        dim: usize,
        hidden_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        qkv_dim: usize,
        n_linear_v_heads: usize,
        seq_len: usize,
        group_size: usize,
    ) -> Result<Self> {
        ensure!(seq_len > 0, "seq_len must be non-zero");
        Ok(Self {
            x: vec![0.0; dim],
            x_normed: vec![0.0; dim],
            xb_normed: vec![0.0; dim],
            xq: QuantizedTensor::new(dim, group_size),
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            hq: QuantizedTensor::new(hidden_dim, group_size),
            q_buf: vec![0.0; 2 * n_heads * head_dim],
            k_buf: vec![0.0; n_kv_heads * head_dim],
            v_buf: vec![0.0; n_kv_heads * head_dim],
            att_buf: vec![0.0; n_heads * seq_len],
            z_buf: vec![0.0; dim],
            qkv_buf: vec![0.0; qkv_dim],
            a_buf: vec![0.0; n_linear_v_heads],
            b_buf: vec![0.0; n_linear_v_heads],
            conv_out: vec![0.0; qkv_dim],
            ffn_out: vec![0.0; dim],
        })
    }
}

// ---------------------------------------------------------------------------
// Top-level Transformer
// ---------------------------------------------------------------------------

pub struct Qwen3_5Transformer {
    config: ModelConfig,
    input_norms: Vec<StandardRMSNorm>,
    post_attention_norms: Vec<StandardRMSNorm>,
    final_norm: StandardRMSNorm,
    embedding_q: QuantizedTensor,
    layers: Vec<LayerWeights>,
    linear_layer_map: Vec<Option<usize>>,
    buffers: TransformerBuffers,
    full_attn_k_cache: Vec<f32>,
    full_attn_v_cache: Vec<f32>,
    delta_conv_states: Vec<Vec<f32>>,
    delta_recurrent_states: Vec<Vec<f32>>,
    logits: Vec<f32>,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    vocab_size: usize,
    group_size: usize,
    seq_len: usize,
    full_attention_mask: u64,
    _mapper: MemoryMapper,
}

impl Qwen3_5Transformer {
    pub(crate) fn new(config: ModelConfig, mut mapper: MemoryMapper) -> Result<Self> {
        let dim = config.dim;
        let n_layers = config.n_layers;
        let n_heads = config.n_heads;
        let n_kv_heads = config.n_kv_heads;
        let head_dim = config.head_dim;
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;
        let group_size = config.group_size;
        let seq_len = config.seq_len;

        let v2 = config.v2.as_ref().ok_or_else(|| anyhow!("Qwen3.5 requires v2 header"))?;
        let norm_eps = v2.norm_eps;
        let rotary_dim = v2.rotary_dim;
        let rope_theta = v2.rope_theta;
        let n_linear_k_heads = v2.n_linear_k_heads;
        let n_linear_v_heads = v2.n_linear_v_heads;
        let linear_k_head_dim = v2.linear_k_head_dim;
        let linear_v_head_dim = v2.linear_v_head_dim;
        let conv_kernel_size = v2.conv_kernel_size;
        let full_attention_mask = v2.full_attention_mask;

        // ── Validation ──
        ensure!(seq_len > 0, "seq_len must be non-zero");
        ensure!(dim.is_multiple_of(group_size), "dim ({}) must be divisible by group_size ({})", dim, group_size);
        ensure!(n_layers > 0, "n_layers must be positive");

        ensure!(n_layers <= 64, "Qwen3.5 v2 supports at most 64 layers, got {n_layers}");
        ensure!(
            n_layers == 64 || full_attention_mask >> n_layers == 0,
            "full_attention_mask has bits set above n_layers"
        );
        let n_full = full_attention_mask.count_ones() as usize;
        ensure!(n_full > 0 && n_full < n_layers, "Qwen3.5 requires both full and linear attention layers");
        let n_linear = n_layers - n_full;

        let kv_dim = n_kv_heads * head_dim;
        let all_heads_dim = n_heads * head_dim;
        ensure!(
            all_heads_dim == dim,
            "Qwen3.5-2B requires n_heads * head_dim ({all_heads_dim}) == hidden size ({dim})"
        );
        let linear_qkv_dim = n_linear_k_heads * linear_k_head_dim
            + n_linear_k_heads * linear_k_head_dim
            + n_linear_v_heads * linear_v_head_dim;

        // Validate target relationships
        ensure!(
            n_heads.is_multiple_of(n_kv_heads),
            "n_heads ({}) must be divisible by n_kv_heads ({})",
            n_heads,
            n_kv_heads
        );
        ensure!(
            n_linear_v_heads == n_linear_k_heads,
            "n_linear_v_heads ({}) must equal n_linear_k_heads ({}) for recurrence",
            n_linear_v_heads,
            n_linear_k_heads
        );
        ensure!(
            linear_v_head_dim == linear_k_head_dim,
            "linear_v_head_dim ({}) must equal linear_k_head_dim ({}) for recurrence",
            linear_v_head_dim,
            linear_k_head_dim
        );
        ensure!(linear_k_head_dim > 0 && linear_v_head_dim > 0, "linear head dims must be positive");
        ensure!(conv_kernel_size > 0, "conv_kernel_size must be positive");
        ensure!(
            rotary_dim > 0 && rotary_dim <= head_dim,
            "rotary_dim ({}) must be between 1 and head_dim ({})",
            rotary_dim,
            head_dim
        );

        // Macro: read f32 vec from mapper
        macro_rules! read_f32v {
            ($count:expr, $label:expr) => {
                mapper.get_f32_slice($count).with_context(|| format!("Failed to read {}", $label))?.to_vec()
            };
        }

        // -- 1. Standard norms (FP32) --
        let raw_input_norms = read_f32v!(n_layers * dim, "input norms");
        let raw_post_norms = read_f32v!(n_layers * dim, "post-attention norms");
        let raw_final_norm = read_f32v!(dim, "final norm");
        let raw_q_norms = read_f32v!(n_layers * head_dim, "Q norms");
        let raw_k_norms = read_f32v!(n_layers * head_dim, "K norms");

        // -- 2. DeltaNet gated norms (FP32) --
        // The exporter emits one fixed-width slot per decoder layer. Full-attention
        // slots contain placeholders and are skipped when constructing Delta layers.
        let raw_gated_norms = read_f32v!(n_layers * linear_v_head_dim, "DeltaNet gated norms");

        // -- 3. FP32 Delta scalars / conv weights --
        let mut a_logs: Vec<Vec<f32>> = Vec::with_capacity(n_linear);
        let mut dt_biases: Vec<Vec<f32>> = Vec::with_capacity(n_linear);
        let mut conv_weights: Vec<Vec<f32>> = Vec::with_capacity(n_linear);
        for i in 0..n_linear {
            a_logs.push(read_f32v!(n_linear_v_heads, &format!("A_log[{}]", i)));
            dt_biases.push(read_f32v!(n_linear_v_heads, &format!("dt_bias[{}]", i)));
            conv_weights.push(read_f32v!(linear_qkv_dim * conv_kernel_size, &format!("conv1d[{}]", i)));
        }

        // -- 4. Quantized tied embedding (Q8) --
        let embed_tensors = read_quantized_tensors(&mut mapper, 1, vocab_size * dim, group_size)?;
        let embedding_q = embed_tensors.into_iter().next().unwrap();

        // -- 5. Per-layer Q8 tensors --
        let mut layers: Vec<LayerWeights> = Vec::with_capacity(n_layers);
        let mut linear_idx = 0usize;

        for layer_idx in 0..n_layers {
            let is_full = (full_attention_mask >> layer_idx) & 1 != 0;

            if is_full {
                let wq_t = read_single_quantized(&mut mapper, dim * 2 * all_heads_dim, group_size)?;
                let wk_t = read_single_quantized(&mut mapper, dim * kv_dim, group_size)?;
                let wv_t = read_single_quantized(&mut mapper, dim * kv_dim, group_size)?;
                let wo_t = read_single_quantized(&mut mapper, all_heads_dim * dim, group_size)?;
                let mlp_w1 = read_single_quantized(&mut mapper, dim * hidden_dim, group_size)?;
                let mlp_w2 = read_single_quantized(&mut mapper, hidden_dim * dim, group_size)?;
                let mlp_w3 = read_single_quantized(&mut mapper, dim * hidden_dim, group_size)?;

                let attn = GatedFullAttention {
                    wq: QLinear { weight: wq_t, in_features: dim, out_features: 2 * all_heads_dim, group_size },
                    wk: QLinear { weight: wk_t, in_features: dim, out_features: kv_dim, group_size },
                    wv: QLinear { weight: wv_t, in_features: dim, out_features: kv_dim, group_size },
                    wo: QLinear { weight: wo_t, in_features: all_heads_dim, out_features: dim, group_size },
                    q_norm: StandardRMSNorm::new(
                        raw_q_norms[layer_idx * head_dim..(layer_idx + 1) * head_dim].to_vec(),
                        norm_eps,
                    ),
                    k_norm: StandardRMSNorm::new(
                        raw_k_norms[layer_idx * head_dim..(layer_idx + 1) * head_dim].to_vec(),
                        norm_eps,
                    ),
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    kv_mul: n_heads / n_kv_heads,
                    rotary_dim,
                    rope_theta,
                    group_size,
                };

                layers.push(LayerWeights::FullAttention(FullAttnLayer {
                    attn,
                    mlp_w1: QLinear { weight: mlp_w1, in_features: dim, out_features: hidden_dim, group_size },
                    mlp_w2: QLinear { weight: mlp_w2, in_features: hidden_dim, out_features: dim, group_size },
                    mlp_w3: QLinear { weight: mlp_w3, in_features: dim, out_features: hidden_dim, group_size },
                }));
            } else {
                let w_qkv_t = read_single_quantized(&mut mapper, dim * linear_qkv_dim, group_size)?;
                let w_z_t = read_single_quantized(&mut mapper, dim * dim, group_size)?;
                let w_b_t = read_single_quantized(&mut mapper, dim * n_linear_v_heads, group_size)?;
                let w_a_t = read_single_quantized(&mut mapper, dim * n_linear_v_heads, group_size)?;
                let w_out_t = read_single_quantized(&mut mapper, dim * dim, group_size)?;
                let mlp_w1 = read_single_quantized(&mut mapper, dim * hidden_dim, group_size)?;
                let mlp_w2 = read_single_quantized(&mut mapper, hidden_dim * dim, group_size)?;
                let mlp_w3 = read_single_quantized(&mut mapper, dim * hidden_dim, group_size)?;

                // Gated norm weight is stored in the slot for this decoder layer.
                let gnorm_weight =
                    raw_gated_norms[layer_idx * linear_v_head_dim..(layer_idx + 1) * linear_v_head_dim].to_vec();

                let delta = GatedDeltaNet {
                    w_qkv: QLinear { weight: w_qkv_t, in_features: dim, out_features: linear_qkv_dim, group_size },
                    w_z: QLinear { weight: w_z_t, in_features: dim, out_features: dim, group_size },
                    w_b: QLinear { weight: w_b_t, in_features: dim, out_features: n_linear_v_heads, group_size },
                    w_a: QLinear { weight: w_a_t, in_features: dim, out_features: n_linear_v_heads, group_size },
                    w_out: QLinear { weight: w_out_t, in_features: dim, out_features: dim, group_size },
                    a_log: a_logs[linear_idx].clone(),
                    dt_bias: dt_biases[linear_idx].clone(),
                    conv_weight: conv_weights[linear_idx].clone(),
                    gated_norm: GatedRMSNorm::new(gnorm_weight, norm_eps, n_linear_v_heads, linear_v_head_dim),
                    n_k_heads: n_linear_k_heads,
                    k_head_dim: linear_k_head_dim,
                    v_head_dim: linear_v_head_dim,
                    q_head_dim: linear_k_head_dim,
                    qkv_dim: linear_qkv_dim,
                    conv_kernel: conv_kernel_size,
                    group_size,
                    dim,
                    norm_eps,
                };

                layers.push(LayerWeights::DeltaNet(DeltaLayer {
                    delta,
                    mlp_w1: QLinear { weight: mlp_w1, in_features: dim, out_features: hidden_dim, group_size },
                    mlp_w2: QLinear { weight: mlp_w2, in_features: hidden_dim, out_features: dim, group_size },
                    mlp_w3: QLinear { weight: mlp_w3, in_features: dim, out_features: hidden_dim, group_size },
                }));

                linear_idx += 1;
            }
        }

        // -- Build layer-index -> linear-cache-index map --
        let mut linear_layer_map = vec![None; n_layers];
        let mut linear_cache_idx = 0usize;
        for (layer_idx, cache_index) in linear_layer_map.iter_mut().enumerate() {
            if (full_attention_mask >> layer_idx) & 1 == 0 {
                *cache_index = Some(linear_cache_idx);
                linear_cache_idx += 1;
            }
        }

        // -- Build norms --
        let input_norms: Vec<_> = (0..n_layers)
            .map(|i| StandardRMSNorm::new(raw_input_norms[i * dim..(i + 1) * dim].to_vec(), norm_eps))
            .collect();
        let post_attention_norms: Vec<_> = (0..n_layers)
            .map(|i| StandardRMSNorm::new(raw_post_norms[i * dim..(i + 1) * dim].to_vec(), norm_eps))
            .collect();
        let final_norm = StandardRMSNorm::new(raw_final_norm, norm_eps);

        // -- Buffers --
        let buffers = TransformerBuffers::new(
            dim,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            linear_qkv_dim,
            n_linear_v_heads,
            seq_len,
            group_size,
        )?;

        // -- Caches --
        let full_attn_k_cache = vec![0.0; n_full * seq_len * kv_dim];
        let full_attn_v_cache = vec![0.0; n_full * seq_len * kv_dim];
        let delta_conv_states = vec![vec![0.0; linear_qkv_dim * conv_kernel_size]; n_linear];
        let delta_recurrent_states =
            vec![vec![0.0; n_linear_v_heads * linear_k_head_dim * linear_v_head_dim]; n_linear];

        let logits = vec![0.0; vocab_size];

        Ok(Self {
            config,
            input_norms,
            post_attention_norms,
            final_norm,
            embedding_q,
            layers,
            linear_layer_map,
            buffers,
            full_attn_k_cache,
            full_attn_v_cache,
            delta_conv_states,
            delta_recurrent_states,
            logits,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            group_size,
            seq_len,
            full_attention_mask,
            _mapper: mapper,
        })
    }

    /// Internal forward step. Returns logits only on the final step (compute_logits=true),
    /// otherwise returns an empty slice to save the LM head computation during prefill.
    fn forward_internal(&mut self, token: usize, pos: usize, compute_logits: bool) -> &[f32] {
        let dim = self.dim;
        let kv_dim = self.n_kv_heads * self.head_dim;
        let all_heads_dim = self.n_heads * self.head_dim;

        // 1. Selected-row embedding
        select_row_dequant(&self.embedding_q, token, dim, self.group_size, &mut self.buffers.x);

        // 2. Process layers
        let mut full_cache_idx = 0usize;

        for layer_idx in 0..self.n_layers {
            let is_full = (self.full_attention_mask >> layer_idx) & 1 != 0;

            // Pre-norm (into scratch, preserve x as residual)
            self.input_norms[layer_idx].forward(&mut self.buffers.x_normed, &self.buffers.x);

            // Quantize normalized input
            quantize(&mut self.buffers.xq, &self.buffers.x_normed, dim, self.group_size);

            // Attention block
            let mut attn_output = vec![0.0; dim];

            if is_full {
                if let LayerWeights::FullAttention(ref fa) = self.layers[layer_idx] {
                    let kv_offset = full_cache_idx * self.seq_len * kv_dim + pos * kv_dim;

                    let out_hd = &mut attn_output[..all_heads_dim];
                    fa.attn.forward(
                        &self.buffers.xq,
                        &mut self.full_attn_k_cache,
                        &mut self.full_attn_v_cache,
                        &mut self.buffers.q_buf,
                        &mut self.buffers.k_buf,
                        &mut self.buffers.v_buf,
                        &mut self.buffers.att_buf,
                        out_hd,
                        pos,
                        kv_offset,
                        self.seq_len,
                    );
                    full_cache_idx += 1;
                }
            } else {
                if let Some(lin_idx) = self.linear_layer_map[layer_idx]
                    && let LayerWeights::DeltaNet(ref dl) = self.layers[layer_idx]
                {
                    dl.delta.forward(
                        &self.buffers.xq,
                        &mut self.buffers.z_buf,
                        &mut self.buffers.qkv_buf,
                        &mut self.buffers.a_buf,
                        &mut self.buffers.b_buf,
                        &mut self.buffers.conv_out,
                        &mut self.delta_conv_states[lin_idx],
                        &mut self.delta_recurrent_states[lin_idx],
                        &mut attn_output,
                        pos,
                    );
                }
            }

            // Residual: x += attn_output (preserve x as residual)
            self.buffers.x.iter_mut().zip(attn_output.iter()).for_each(|(x, &attention)| *x += attention);

            // Post-norm (into xb_normed scratch, preserve x as residual)
            self.post_attention_norms[layer_idx].forward(&mut self.buffers.xb_normed, &self.buffers.x);

            // Quantize normalized MLP input
            quantize(&mut self.buffers.xq, &self.buffers.xb_normed, dim, self.group_size);

            // SwiGLU MLP
            let (w1, w3, w2) = match &self.layers[layer_idx] {
                LayerWeights::FullAttention(fa) => (&fa.mlp_w1, &fa.mlp_w3, &fa.mlp_w2),
                LayerWeights::DeltaNet(dl) => (&dl.mlp_w1, &dl.mlp_w3, &dl.mlp_w2),
            };

            w1.forward(&mut self.buffers.hb, &self.buffers.xq);
            w3.forward(&mut self.buffers.hb2, &self.buffers.xq);

            for i in 0..self.hidden_dim {
                let gate = self.buffers.hb[i];
                self.buffers.hb[i] = gate * (1.0 + (-gate).exp()).recip() * self.buffers.hb2[i];
            }

            quantize(&mut self.buffers.hq, &self.buffers.hb, self.hidden_dim, self.group_size);
            w2.forward(&mut self.buffers.ffn_out, &self.buffers.hq);

            // Residual: x += ffn_out
            for i in 0..dim {
                self.buffers.x[i] += self.buffers.ffn_out[i];
            }
        }

        // 3. Final norm
        self.final_norm.forward_inplace(&mut self.buffers.x);

        // 4. LM head (tied embedding, matmul) — only when compute_logits is true
        if compute_logits {
            quantize(&mut self.buffers.xq, &self.buffers.x, dim, self.group_size);
            matmul(&mut self.logits, &self.buffers.xq, &self.embedding_q, dim, self.vocab_size, self.group_size);
            &self.logits
        } else {
            // Return empty slice during prefill for non-final tokens
            &[]
        }
    }

    pub fn forward(&mut self, token: usize, pos: usize) -> &[f32] {
        // Always compute logits for single-token forward (used by generation)
        self.forward_internal(token, pos, true)
    }

    pub fn reset_cache_impl(&mut self) {
        self.full_attn_k_cache.fill(0.0);
        self.full_attn_v_cache.fill(0.0);
        for state in self.delta_conv_states.iter_mut() {
            state.fill(0.0);
        }
        for state in self.delta_recurrent_states.iter_mut() {
            state.fill(0.0);
        }
    }

    pub fn prefill_impl(&mut self, tokens: &[usize]) -> &[f32] {
        if tokens.is_empty() {
            return &[];
        }
        self.reset_cache_impl();
        let (&last_token, prefix) = tokens.split_last().expect("tokens is non-empty");
        // Non-final tokens: skip LM head
        for (pos, &token) in prefix.iter().enumerate() {
            self.forward_internal(token, pos, false);
        }
        // Final token: compute logits
        self.forward_internal(last_token, prefix.len(), true)
    }

    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Helpers: loading quantized tensors
// ---------------------------------------------------------------------------

fn read_quantized_tensors(
    mapper: &mut MemoryMapper,
    count: usize,
    size_each: usize,
    group_size: usize,
) -> Result<Vec<QuantizedTensor>> {
    (0..count).map(|_| read_single_quantized(mapper, size_each, group_size)).collect()
}

fn read_single_quantized(mapper: &mut MemoryMapper, total_elems: usize, group_size: usize) -> Result<QuantizedTensor> {
    let q_bytes = mapper.get_bytes(total_elems).context("Failed to read quantized tensor data")?;
    let q_slice = unsafe { std::slice::from_raw_parts(q_bytes.as_ptr() as *const i8, total_elems) };
    let s_len = total_elems / group_size;
    let s_slice = mapper.get_f32_slice(s_len).context("Failed to read quantized tensor scales")?;
    let q_static = unsafe { std::mem::transmute::<&[i8], &'static [i8]>(q_slice) };
    let s_static = unsafe { std::mem::transmute::<&[f32], &'static [f32]>(s_slice) };
    Ok(QuantizedTensor::from_slices(q_static, s_static))
}

// ---------------------------------------------------------------------------
// Selected-row embedding dequantization
// ---------------------------------------------------------------------------

fn select_row_dequant(q: &QuantizedTensor, row: usize, dim: usize, group_size: usize, output: &mut [f32]) {
    for (col, out) in output.iter_mut().enumerate().take(dim) {
        let idx = row * dim + col;
        let group_idx = idx / group_size;
        *out = q.q[idx] as f32 * q.s[group_idx];
    }
}
